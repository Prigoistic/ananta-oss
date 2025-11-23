# Contributing to Ananta

Thank you for your interest in contributing to Ananta! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a new branch for your feature
4. Make your changes
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ananta-update.git
cd ananta-update

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements.txt
```

## Project Structure Guidelines

- **src/training/** - Training-related scripts and modules
- **src/data/** - Data processing and preparation
- **src/evaluation/** - Model evaluation and metrics
- **src/utils/** - Shared utility functions
- **configs/** - Configuration files
- **demos/** - Demo applications
- **tests/** - Test scripts and unit tests
- **deployment/** - Deployment configurations

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Include docstrings for functions and classes
- Add comments for complex logic
- Keep functions focused and modular

## Adding New Features

### Training Features

Add new training methods in `src/training/`:
- Create a new file for major features
- Update `__init__.py` to export new functions
- Add configuration options to `configs/train_config.json`

### Data Processing

Add new data processors in `src/data/`:
- Support for new dataset formats
- Additional preprocessing steps
- Data augmentation techniques

### Evaluation Metrics

Add new evaluation methods in `src/evaluation/`:
- New metrics for model performance
- Visualization tools
- Benchmark comparisons

## Testing

Before submitting a PR:

1. Test your changes locally
2. Ensure backward compatibility
3. Add tests if applicable
4. Update documentation

## Documentation

- Update README.md for major changes
- Add docstrings to new functions/classes
- Update configuration documentation
- Add examples for new features

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the version number in `src/__init__.py` following semantic versioning
3. Ensure all tests pass
4. Get at least one code review approval
5. The PR will be merged once approved

## Commit Message Guidelines

- Use clear, descriptive commit messages
- Start with a verb in present tense (Add, Fix, Update, etc.)
- Reference issues when applicable

Examples:
```
Add support for custom dataset formats
Fix memory leak in training loop
Update evaluation metrics documentation
```

## Areas for Contribution

### High Priority
- Performance optimization
- Memory efficiency improvements
- Additional evaluation metrics
- Better error handling
- More comprehensive tests

### Medium Priority
- Additional dataset format support
- Visualization improvements
- Documentation enhancements
- Example notebooks

### Future Features
- Distributed training support
- Multi-GPU optimization
- API server implementation
- Web-based training dashboard

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Focus on what is best for the community

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about implementation
- Documentation clarifications

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing to Ananta! 🎉
