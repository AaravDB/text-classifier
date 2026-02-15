"""
Main entry point for the project.

Runs all experiments.
"""

from src.experiment import run_all_experiments


def main():

    print("Starting text classification experiments...\n")

    results = run_all_experiments()

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
