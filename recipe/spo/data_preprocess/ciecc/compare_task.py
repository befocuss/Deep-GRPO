import os
import json
from typing import Dict, Any, List

from datasets import Dataset


RAW_JSON_PATH = "/data/download/ciecc/compare.json"
WORKSPACE = "/data/download/ciecc/data"
SAVE_PATH = "/data/hf-datasets/ciecc/compare"
DATA_SOURCE_NAME = "ciecc_compare"

SYSTEM_PROMPT = """
你是一个面向政府与企业的数据分析与算法选型专家。

你可以使用的算法库包含以下算法（算法名在后续会与接口文档一一对应）：
- Adaboost
- Apriori算法
- BP神经网络模型
- KNN
- K均值聚类分析
- logistic回归
- Pearson
- Spearman
- TOPSIS法
- T检验
- 主成分分析
- 决策树
- 净现值法
- 判别分析
- 区位熵分析
- 因子分析
- 多项式回归
- 层次分析法
- 岭回归
- 差分自回归移动平均模型（ARIMA）
- 弹性系数法
- 指数平滑法
- 支持向量机分类
- 支持向量机回归
- 数据包络分析法
- 最小二乘回归
- 李克特量表
- 梯度提升树
- 模糊综合评价
- 正态检验
- 滑动平均模型（MA）
- 灰色模型
- 秩和比评价法
- 移动平均法
- 自回归模型（AR）
- 自回归滑动平均模型（ARMA）
- 蒙特卡洛模拟
- 道格拉斯生产函数
- 随机森林

你无法直接看到这些算法的接口文档，但可以通过下面的方式逐个检索算法文档：
1. 使用标签 <Retrieve>算法名</Retrieve> 发起一次检索。
   - 每次只能传入一个算法名。
2. 系统会返回该算法的接口文档，并用 <Result>...</Result> 包裹。
3. 如果你需要多个算法的接口文档，需要多次使用 <Retrieve>算法名</Retrieve>，依次获取。

你的任务是：
1. 阅读用户给出的数据分析需求（来自 user 角色的自然语言问题）。
2. 根据需求，选择合适的算法；如有需要，通过 <Retrieve>算法名</Retrieve> 获取相应的算法接口文档。
3. 使用获取到的算法接口文档，进行详细的数据分析，完成用户提出的数据分析问题。
4. 对于需要多种算法比较的场景，可以多次使用 <Retrieve>，分别获取各算法文档后进行对比与分析。

""".strip()


USER_PROMPT_TEMPLATE = """
{query}

数据文件：
{dataset_name}
"""

def build_single_verl_example(
    *,
    dataset_name: str,
    family: str,
    question: Dict[str, Any],
) -> Dict[str, Any]:
    query = question["user_query"]
    judge_rubric = question["judge_rubric"]
    candidate_algorithms: List[str] = question["candidate_algorithms"]

    user_query = USER_PROMPT_TEMPLATE.format(
        query=query,
        dataset_name=dataset_name,
    )

    data = {
        "data_source": DATA_SOURCE_NAME,
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        "ability": "ciecc_compare",
        "reward_model": {
            "style": "rule",
            "ground_truth": judge_rubric,
        },
        "extra_info": {
            "query": query,
            "dataset": dataset_name,
            "family": family,
            "candidate_algorithms": candidate_algorithms,
            "workspace": WORKSPACE,
        }
    }
    return data


def main():
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    flat_examples: List[Dict[str, Any]] = []

    for family_name, datasets in raw.items():
        for ds in datasets:
            is_applicable = ds["is_applicable"]
            if not is_applicable:
                continue

            dataset_name = ds["dataset"]
            family = ds["family"]

            questions = ds["questions"]
            assert len(ds["questions"]) > 0

            for question in questions:
                row = build_single_verl_example(
                    dataset_name=dataset_name,
                    family=family,
                    question=question
                )
                if row is not None:
                    flat_examples.append(row)

    print(f"Total questions after filter (is_applicable=True) & flatten: {len(flat_examples)}")

    hf_dataset = Dataset.from_list(flat_examples)
    hf_dataset = hf_dataset.shuffle(seed=42)

    split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    print(
        f"Split sizes -> Train: {len(split_dataset['train'])}, "
        f"Test: {len(split_dataset['test'])}"
    )

    os.makedirs(SAVE_PATH, exist_ok=True)
    train_path = os.path.join(SAVE_PATH, "train.parquet")
    test_path = os.path.join(SAVE_PATH, "test.parquet")

    split_dataset["train"].to_parquet(train_path)
    split_dataset["test"].to_parquet(test_path)

    print(f"Saved train to {train_path}")
    print(f"Saved test to {test_path}")


if __name__ == "__main__":
    main()
