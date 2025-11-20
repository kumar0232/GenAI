import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tabulate import tabulate
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AgentTester:
    def __init__(
        self, api_url: str, ollama_url: str, model_name: str, verbose: bool = False
    ):
        self.api_url = api_url.replace("localhost", "127.0.0.1")  # Force IPv4
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.verbose = verbose
        self.session = requests.Session()
        # Configure retries
        retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.test_cases = self._load_test_cases()
        self.conversations = {}
        self.results = []

    def _load_test_cases(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load test cases from JSON file"""
        file_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
        logger.info(f"Loading test cases from: {file_path}")
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load test cases: {e}")
            raise

    def _evaluate_response(
        self,
        query: str,
        actual_response: str,
        expected_output: str,
        criteria: Dict[str, Any],
        agent_type: str,
    ) -> Dict[str, Any]:
        """Use the Groq API to evaluate a response against expected criteria"""
        prompt = f"""
        You are evaluating an AI agent's response to a customer support query.

        QUERY:
        {query}

        ACTUAL RESPONSE:
        {actual_response}

        EXPECTED PATTERNS:
        {expected_output}

        Please evaluate the response based on the following criteria:
        1. Accuracy (0-10): Does the response contain correct information?
        2. Completeness (0-10): Does the response address all aspects of the query?
        3. Relevance (0-10): Is the response focused on answering the specific question?
        4. Clarity (0-10): Is the response clear and easy to understand?
        5. Agent appropriateness (0-10): Was this handled by the appropriate agent type? The system identified it as "{agent_type}".

        Respond with a JSON object containing:
        {{
            "accuracy": score,
            "completeness": score,
            "relevance": score,
            "clarity": score,
            "agent_appropriateness": score,
            "total_score": sum_of_scores,
            "percentage": percentage_of_max_possible,
            "strengths": ["list", "of", "strengths"],
            "weaknesses": ["list", "of", "weaknesses"],
            "passed": true/false
        }}
        
        Only respond with the JSON, nothing else.
        """

        try:
            from groq import Groq
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))
            response = groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an evaluator for customer support responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            llm_response = response.choices[0].message.content

            # Parse the JSON response
            try:
                json_str = llm_response.strip()
                if not json_str.startswith("{"):
                    start_idx = llm_response.find("{")
                    end_idx = llm_response.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = llm_response[start_idx:end_idx]
                evaluation = json.loads(json_str)
                return evaluation
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse evaluation response: {e}\nResponse: {llm_response}")
                return {
                    "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "clarity": 0,
                    "agent_appropriateness": 0,
                    "total_score": 0,
                    "percentage": 0,
                    "strengths": [],
                    "weaknesses": ["Failed to evaluate response"],
                    "passed": False,
                }
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {
                "accuracy": 0,
                "completeness": 0,
                "relevance": 0,
                "clarity": 0,
                "agent_appropriateness": 0,
                "total_score": 0,
                "percentage": 0,
                "strengths": [],
                "weaknesses": [f"Evaluation error: {str(e)}"],
                "passed": False,
            }

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case and evaluate the response"""
        query = test_case["query"]
        expected_output = test_case["expected_output"]
        category = test_case["category"]
        test_id = test_case["id"]
        criteria = test_case.get("criteria", {})

        if test_id not in self.conversations:
            self.conversations[test_id] = f"test-{test_id}-{int(time.time())}"

        try:
            logger.info(f"Sending request to: {self.api_url}/api/query")
            logger.info(f"Request payload: {{'query': {query}, 'conversation_id': {self.conversations[test_id]}}}")
            response = self.session.post(
                f"{self.api_url}/api/query",
                json={"query": query, "conversation_id": self.conversations[test_id]},
                timeout=10  # Increased timeout
            )
            response.raise_for_status()

            api_response = response.json()
            actual_response = api_response.get("response", "")
            agent_type = api_response.get("agent", "unknown")

            if self.verbose:
                logger.info(f"Test ID: {test_id}")
                logger.info(f"Query: {query}")
                logger.info(f"Response (from {agent_type}):\n{actual_response}")

            evaluation = self._evaluate_response(
                query, actual_response, expected_output, criteria, agent_type
            )

            result = {
                "id": test_id,
                "category": category,
                "query": query,
                "expected_output": expected_output,
                "actual_response": actual_response,
                "agent_type": agent_type,
                "evaluation": evaluation,
            }

            return result
        except Exception as e:
            logger.error(f"Error running test {test_id}: {e}")
            return {
                "id": test_id,
                "category": category,
                "query": query,
                "expected_output": expected_output,
                "actual_response": f"Error: {str(e)}",
                "agent_type": "error",
                "evaluation": {
                    "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "clarity": 0,
                    "agent_appropriateness": 0,
                    "total_score": 0,
                    "percentage": 0,
                    "strengths": [],
                    "weaknesses": [f"Test execution error: {str(e)}"],
                    "passed": False,
                },
            }

    def run_tests(self, categories: List[str] = None, concurrent: bool = False):
        """Run all test cases, optionally filtering by category"""
        all_tests = []

        for category, tests in self.test_cases.items():
            if categories is None or category in categories:
                for test in tests:
                    test["category"] = category
                    all_tests.append(test)

        logger.info(f"Running {len(all_tests)} tests...")

        if concurrent:
            with ThreadPoolExecutor(max_workers=5) as executor:
                self.results = list(executor.map(self.run_test, all_tests))
        else:
            self.results = [self.run_test(test) for test in all_tests]

        logger.info("Testing completed")

    def generate_report(self, output_path: str = "test_results"):
        """Generate a comprehensive test report"""
        if not self.results:
            logger.error("No test results available. Run tests first.")
            return

        os.makedirs(output_path, exist_ok=True)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["evaluation"]["passed"])
        pass_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0

        avg_scores = {
            "accuracy": sum(r["evaluation"]["accuracy"] for r in self.results) / total_tests,
            "completeness": sum(r["evaluation"]["completeness"] for r in self.results) / total_tests,
            "relevance": sum(r["evaluation"]["relevance"] for r in self.results) / total_tests,
            "clarity": sum(r["evaluation"]["clarity"] for r in self.results) / total_tests,
            "agent_appropriateness": sum(r["evaluation"]["agent_appropriateness"] for r in self.results) / total_tests,
            "overall": sum(r["evaluation"]["percentage"] for r in self.results) / total_tests,
        }

        category_results = {}
        for result in self.results:
            category = result["category"]
            if category not in category_results:
                category_results[category] = {"total": 0, "passed": 0, "score": 0}

            category_results[category]["total"] += 1
            if result["evaluation"]["passed"]:
                category_results[category]["passed"] += 1
            category_results[category]["score"] += result["evaluation"]["percentage"]

        for category in category_results:
            total = category_results[category]["total"]
            if total > 0:
                category_results[category]["pass_rate"] = (
                    category_results[category]["passed"] / total * 100
                )
                category_results[category]["avg_score"] = (
                    category_results[category]["score"] / total
                )
            else:
                category_results[category]["pass_rate"] = 0
                category_results[category]["avg_score"] = 0

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "pass_rate": pass_rate,
                "average_scores": avg_scores,
            },
            "category_results": category_results,
            "test_results": self.results,
        }

        with open(os.path.join(output_path, "test_results.json"), "w") as f:
            json.dump(report, f, indent=2)

        with open(os.path.join(output_path, "summary_report.md"), "w") as f:
            f.write("# Automated Test Results Summary\n\n")
            f.write("## Overall Summary\n\n")
            f.write(f"- Total Tests: {total_tests}\n")
            f.write(f"- Passed Tests: {passed_tests}\n")
            f.write(f"- Pass Rate: {pass_rate:.2f}%\n\n")
            f.write("## Average Scores\n\n")
            f.write(f"- Accuracy: {avg_scores['accuracy']:.2f}/10\n")
            f.write(f"- Completeness: {avg_scores['completeness']:.2f}/10\n")
            f.write(f"- Relevance: {avg_scores['relevance']:.2f}/10\n")
            f.write(f"- Clarity: {avg_scores['clarity']:.2f}/10\n")
            f.write(f"- Agent Appropriateness: {avg_scores['agent_appropriateness']:.2f}/10\n")
            f.write(f"- Overall Score: {avg_scores['overall']:.2f}%\n\n")
            f.write("## Results by Category\n\n")

            categories_table = []
            for category, results in category_results.items():
                categories_table.append(
                    [
                        category,
                        f"{results['passed']}/{results['total']}",
                        f"{results['pass_rate']:.2f}%",
                        f"{results['avg_score']:.2f}%",
                    ]
                )

            f.write(
                tabulate(
                    categories_table,
                    headers=["Category", "Passed/Total", "Pass Rate", "Avg Score"],
                    tablefmt="pipe",
                )
            )
            f.write("\n\n")

            f.write("## Failed Tests\n\n")
            failed_tests = [r for r in self.results if not r["evaluation"]["passed"]]
            if failed_tests:
                for i, test in enumerate(failed_tests, 1):
                    f.write(f"### {i}. Test ID: {test['id']} ({test['category']})\n\n")
                    f.write(f"**Query:** {test['query']}\n\n")
                    f.write(f"**Agent:** {test['agent_type']}\n\n")
                    f.write("**Weaknesses:**\n")
                    for weakness in test["evaluation"]["weaknesses"]:
                        f.write(f"- {weakness}\n")
                    f.write("\n")
                    f.write(f"**Score:** {test['evaluation']['percentage']:.2f}%\n\n")
                    f.write("---\n\n")
            else:
                f.write("No failed tests! 🎉\n\n")

        self._generate_visual_report(output_path, report)
        logger.info(f"Report generated in {output_path}")
        return report["summary"]["average_scores"]["overall"]

    def _generate_visual_report(self, output_path: str, report: Dict[str, Any]):
        """Generate visual representations of test results"""
        try:
            categories = list(report["category_results"].keys())
            pass_rates = [
                report["category_results"][c]["pass_rate"] for c in categories
            ]
            avg_scores = [
                report["category_results"][c]["avg_score"] for c in categories
            ]

            plt.figure(figsize=(10, 6))
            x = range(len(categories))
            width = 0.35

            plt.bar(
                [i - width / 2 for i in x], pass_rates, width, label="Pass Rate (%)"
            )
            plt.bar(
                [i + width / 2 for i in x], avg_scores, width, label="Avg Score (%)"
            )

            plt.xlabel("Category")
            plt.ylabel("Percentage")
            plt.title("Test Results by Category")
            plt.xticks(x, categories, rotation=45)
            plt.legend()
            plt.tight_layout()

            plt.savefig(os.path.join(output_path, "category_results.png"))
            plt.close()

            score_categories = [
                "Accuracy",
                "Completeness",
                "Relevance",
                "Clarity",
                "Agent Appropriateness",
            ]
            score_values = [
                report["summary"]["average_scores"]["accuracy"] / 10 * 100,
                report["summary"]["average_scores"]["completeness"] / 10 * 100,
                report["summary"]["average_scores"]["relevance"] / 10 * 100,
                report["summary"]["average_scores"]["clarity"] / 10 * 100,
                report["summary"]["average_scores"]["agent_appropriateness"] / 10 * 100,
            ]

            angles = np.linspace(
                0, 2 * np.pi, len(score_categories), endpoint=False
            ).tolist()
            angles += angles[:1]
            score_values += score_values[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.plot(angles, score_values, "o-", linewidth=2)
            ax.fill(angles, score_values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), score_categories)
            ax.set_ylim(0, 100)
            ax.grid(True)
            ax.set_title("Evaluation Criteria Scores", size=20, y=1.05)

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "evaluation_radar.png"))
            plt.close()

        except Exception as e:
            logger.error(f"Error generating visual report: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Automated testing for TechSolutions Support Agent Orchestrator"
    )
    parser.add_argument(
        "--api-url", default="http://127.0.0.1:8000", help="URL of the agent API"
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="URL of the Ollama API"
    )
    parser.add_argument(
        "--model", default="llama3-8b-8192", help="LLM model to use for evaluation"
    )
    parser.add_argument("--categories", nargs="+", help="Categories of tests to run")
    parser.add_argument(
        "--output", default="test_results", help="Output directory for test results"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--concurrent", action="store_true", help="Run tests concurrently"
    )

    args = parser.parse_args()

    tester = AgentTester(args.api_url, args.ollama_url, args.model, args.verbose)
    tester.run_tests(args.categories, args.concurrent)
    overall_score = tester.generate_report(args.output)

    print(f"\nTesting completed. Overall score: {overall_score:.2f}%")
    print(f"Detailed results available in: {args.output}")

if __name__ == "__main__":
    main()