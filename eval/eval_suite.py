import requests
import json
import time
import pandas as pd
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "How much did I spend on food in total?",
        "expected_contains": ["food", "kes", "spend"],
        "intent": "question",
        "category": "spending_total"
    },
    {
        "id": 2,
        "question": "What is my most expensive transaction?",
        "expected_contains": ["kes", "transaction"],
        "intent": "question",
        "category": "single_transaction"
    },
    {
        "id": 3,
        "question": "How much did I spend on transport?",
        "expected_contains": ["transport", "kes"],
        "intent": "question",
        "category": "spending_total"
    },
    {
        "id": 4,
        "question": "What are my top 3 spending categories?",
        "expected_contains": ["food", "rent", "kes"],
        "intent": "question",
        "category": "category_breakdown"
    },
    {
        "id": 5,
        "question": "Are there any suspicious transactions?",
        "expected_contains": ["transaction", "suspicious", "unusual"],
        "intent": "anomaly",
        "category": "anomaly_detection"
    },
    {
        "id": 6,
        "question": "What will I spend on food next month?",
        "expected_contains": ["food", "forecast", "kes"],
        "intent": "forecast",
        "category": "forecasting"
    },
    {
        "id": 7,
        "question": "How much did I spend at Naivas Supermarket?",
        "expected_contains": ["naivas", "kes"],
        "intent": "question",
        "category": "merchant_specific"
    },
    {
        "id": 8,
        "question": "Did I spend more on food or transport?",
        "expected_contains": ["food", "transport", "kes"],
        "intent": "question",
        "category": "comparison"
    },
    {
        "id": 9,
        "question": "Which transactions look unusual this week?",
        "expected_contains": ["transaction"],
        "intent": "anomaly",
        "category": "anomaly_detection"
    },
    {
        "id": 10,
        "question": "What is my average daily spending?",
        "expected_contains": ["kes", "average", "daily"],
        "intent": "question",
        "category": "spending_total"
    },
    {
        "id": 11,
        "question": "How much did I spend on entertainment?",
        "expected_contains": ["entertainment", "kes"],
        "intent": "question",
        "category": "spending_total"
    },
    {
        "id": 12,
        "question": "What will my transport costs be next month?",
        "expected_contains": ["transport", "kes"],
        "intent": "forecast",
        "category": "forecasting"
    },
    {
        "id": 13,
        "question": "Show me my spending breakdown by category",
        "expected_contains": ["food", "transport", "kes"],
        "intent": "classify",
        "category": "category_breakdown"
    },
    {
        "id": 14,
        "question": "What was my biggest food purchase?",
        "expected_contains": ["food", "kes"],
        "intent": "question",
        "category": "single_transaction"
    },
    {
        "id": 15,
        "question": "How much did I spend on health and pharmacy?",
        "expected_contains": ["health", "kes"],
        "intent": "question",
        "category": "spending_total"
    },
    {
        "id": 16,
        "question": "Are there any high severity anomalies?",
        "expected_contains": ["transaction", "suspicious"],
        "intent": "anomaly",
        "category": "anomaly_detection"
    },
    {
        "id": 17,
        "question": "What merchant do I spend the most at?",
        "expected_contains": ["kes", "merchant"],
        "intent": "question",
        "category": "merchant_specific"
    },
    {
        "id": 18,
        "question": "How much did I spend on weekends vs weekdays?",
        "expected_contains": ["kes", "spend"],
        "intent": "question",
        "category": "comparison"
    },
    {
        "id": 19,
        "question": "Give me a summary of my total spending",
        "expected_contains": ["kes", "total", "spend"],
        "intent": "question",
        "category": "spending_total"
    },
    {
        "id": 20,
        "question": "What is my shopping budget looking like next month?",
        "expected_contains": ["shopping", "kes"],
        "intent": "forecast",
        "category": "forecasting"
    }
]


def run_single_test(test: dict) -> dict:
    start = time.time()

    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": test["question"]},
            timeout=60
        )
        elapsed = (time.time() - start) * 1000

        if r.status_code != 200:
            return {
                **test,
                "passed": False,
                "answer": f"HTTP {r.status_code}",
                "actual_intent": "error",
                "intent_correct": False,
                "keywords_found": [],
                "keywords_missing": test["expected_contains"],
                "response_time_ms": elapsed,
                "error": r.text
            }

        data = r.json()
        answer = data.get("answer", "").lower()
        actual_intent = data.get("intent", "")

        keywords_found = [k for k in test["expected_contains"] if k.lower() in answer]
        keywords_missing = [k for k in test["expected_contains"] if k.lower() not in answer]

        keyword_score = len(keywords_found) / len(test["expected_contains"])
        intent_correct = actual_intent == test["intent"]
        passed = keyword_score >= 0.5 and len(answer) > 20

        return {
            **test,
            "passed": passed,
            "answer": data.get("answer", ""),
            "actual_intent": actual_intent,
            "intent_correct": intent_correct,
            "keyword_score": round(keyword_score, 2),
            "keywords_found": keywords_found,
            "keywords_missing": keywords_missing,
            "response_time_ms": round(elapsed, 2),
            "error": None
        }

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return {
            **test,
            "passed": False,
            "answer": "",
            "actual_intent": "error",
            "intent_correct": False,
            "keyword_score": 0,
            "keywords_found": [],
            "keywords_missing": test["expected_contains"],
            "response_time_ms": round(elapsed, 2),
            "error": str(e)
        }


def run_eval_suite() -> dict:
    print("\n" + "="*60)
    print("FINSIGHT AGENT — EVALUATION SUITE")
    print("="*60)
    print(f"Running {len(TEST_QUESTIONS)} test questions...\n")

    results = []

    for test in TEST_QUESTIONS:
        print(f"Testing Q{test['id']}: {test['question'][:55]}...")
        result = run_single_test(test)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {status} | Intent: {result['actual_intent']} | "
              f"Keywords: {result['keyword_score']:.0%} | "
              f"Time: {result['response_time_ms']:.0f}ms")

    df = pd.DataFrame(results)

    total = len(df)
    passed = df['passed'].sum()
    overall_accuracy = passed / total
    avg_response_time = df['response_time_ms'].mean()
    p95_response_time = df['response_time_ms'].quantile(0.95)
    intent_accuracy = df['intent_correct'].mean()
    avg_keyword_score = df['keyword_score'].mean()

    by_category = df.groupby('category')['passed'].agg(['sum', 'count'])
    by_category['accuracy'] = by_category['sum'] / by_category['count']

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total,
        "passed": int(passed),
        "failed": int(total - passed),
        "overall_accuracy": round(overall_accuracy, 4),
        "intent_accuracy": round(intent_accuracy, 4),
        "avg_keyword_score": round(avg_keyword_score, 4),
        "avg_response_time_ms": round(avg_response_time, 2),
        "p95_response_time_ms": round(p95_response_time, 2),
        "by_category": by_category.to_dict()
    }

    df.to_csv("eval/eval_results.csv", index=False)

    with open("eval/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"  Overall accuracy     : {overall_accuracy:.1%} ({passed}/{total})")
    print(f"  Intent accuracy      : {intent_accuracy:.1%}")
    print(f"  Avg keyword score    : {avg_keyword_score:.1%}")
    print(f"  Avg response time    : {avg_response_time:.0f}ms")
    print(f"  P95 response time    : {p95_response_time:.0f}ms")

    print("\nBy category:")
    for cat, row in by_category.iterrows():
        print(f"  {cat:<25} {row['accuracy']:.0%} ({int(row['sum'])}/{int(row['count'])})")

    failed_tests = df[df['passed'] == False]
    if len(failed_tests) > 0:
        print(f"\nFailed tests ({len(failed_tests)}):")
        for _, row in failed_tests.iterrows():
            print(f"  Q{row['id']}: {row['question'][:50]}...")
            print(f"    Missing keywords: {row['keywords_missing']}")

    print("\nResults saved to eval/eval_results.csv")
    print("Summary saved to eval/eval_summary.json")

    return summary


if __name__ == "__main__":
    summary = run_eval_suite()

    print("\n" + "="*60)
    if summary['overall_accuracy'] >= 0.75:
        print(f"EVALUATION PASSED — {summary['overall_accuracy']:.1%} accuracy")
    else:
        print(f"NEEDS IMPROVEMENT — {summary['overall_accuracy']:.1%} accuracy (target: 75%+)")
    print("="*60)