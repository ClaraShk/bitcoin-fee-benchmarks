# Contributing to Bitcoin Fee Benchmarks

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Ways to Contribute

### 1. Share Your Results

The easiest way to contribute is to share your predictor results:

1. **Create your predictor** following the [Predictor Guide](docs/PREDICTOR_GUIDE.md)
2. **Evaluate it:** `python evaluate.py your_predictor.py --horizon 3h`
3. **Share results** via:
   - GitHub Issue using the "Share Results" template
   - GitHub Discussions in "Show & Tell"

Include:
- Brief description of your approach
- Results JSON file
- Key insights or observations

### 2. Submit a Predictor

Add your predictor to the `examples/` directory:

1. Fork the repository
2. Create a branch: `git checkout -b add-my-predictor`
3. Add your predictor: `examples/my_predictor.py`
4. Ensure it follows the `BasePredictor` interface
5. Add a docstring explaining your approach
6. Test with both horizons: `--horizon 3h` and `--horizon 1d`
7. Submit a PR

### 3. Improve Documentation

- Fix typos or unclear explanations
- Add examples or tutorials
- Improve docstrings
- Translate documentation

### 4. Report Issues

Found a bug or have a suggestion? Open an issue with:
- Clear description
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (Python version, OS)

### 5. Add Features

Want to add new features? Great! Please:
- Open an issue first to discuss
- Follow the code style guidelines
- Add tests if applicable
- Update documentation

## Code Style

### Python

- Follow PEP 8
- Use type hints for function signatures
- Include docstrings for public functions
- Maximum line length: 100 characters

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: int, param2: str) -> float:
    """
    Brief description of function.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Example:
        >>> my_function(1, "test")
        0.5
    """
```

### Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Be descriptive but concise
- Reference issues when applicable: "Fix #123"

Examples:
- `Add gradient boosting predictor`
- `Fix timezone handling in evaluation`
- `Update README with LLM guidance`

## Pull Request Process

1. **Fork and branch:** Create a feature branch from `main`

2. **Make changes:** Implement your feature or fix

3. **Test:** Ensure everything works
   ```bash
   # Run basic tests
   python evaluate.py examples/template_predictor.py --horizon 3h --max-snapshots 10
   ```

4. **Commit:** Use clear commit messages

5. **Push:** Push to your fork

6. **PR:** Open a pull request with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/results if applicable

7. **Review:** Address any feedback

## Testing Guidelines

### For Predictors

```bash
# Test basic functionality
PYTHONPATH=. python examples/my_predictor.py

# Run evaluation
python evaluate.py examples/my_predictor.py --horizon 3h --max-snapshots 50

# Compare with baselines
python scripts/compare_results.py results/*.json
```

### For Core Changes

Ensure existing functionality still works:

```bash
# Test data loading
PYTHONPATH=. python -c "from benchmarks import load_dataset; print('OK')"

# Test evaluation
python evaluate.py examples/naive_predictor.py --horizon 3h --max-snapshots 10 --no-save
```

## Repository Structure

```
bitcoin-fee-benchmarks/
├── benchmarks/        # Core library (be careful modifying)
├── data/              # Datasets (don't commit changes)
├── docs/              # Documentation (PRs welcome)
├── examples/          # Predictors (add yours here!)
├── notebooks/         # Jupyter notebooks
├── results/           # Evaluation results
├── scripts/           # Utility scripts
└── .github/           # GitHub templates
```

## Questions?

- Check existing issues and discussions
- Open a new discussion for questions
- Tag maintainers if urgent

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing!
