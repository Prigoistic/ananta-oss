"""
Logging utilities for HMTT.

Provides structured logging for training, evaluation, and inference.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "HMTT",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_training_logger(log_dir: str = "logs") -> logging.Logger:
    """
    Setup logger for training.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"training_{timestamp}.log"
    
    return get_logger(
        name="HMTT.Training",
        level=logging.INFO,
        log_file=str(log_file)
    )


def setup_evaluation_logger(log_dir: str = "logs") -> logging.Logger:
    """
    Setup logger for evaluation.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"evaluation_{timestamp}.log"
    
    return get_logger(
        name="HMTT.Evaluation",
        level=logging.INFO,
        log_file=str(log_file)
    )


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = ""):
    """
    Log metrics dictionary.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        prefix: Optional prefix for metric names
    """
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, float):
            logger.info(f"{metric_name}: {value:.4f}")
        else:
            logger.info(f"{metric_name}: {value}")


def log_training_step(
    logger: logging.Logger,
    step: int,
    total_steps: int,
    metrics: dict
):
    """
    Log training step information.
    
    Args:
        logger: Logger instance
        step: Current step
        total_steps: Total steps
        metrics: Step metrics
    """
    logger.info(f"Step {step}/{total_steps}")
    log_metrics(logger, metrics, prefix="  ")


def log_epoch(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    metrics: dict
):
    """
    Log epoch information.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        total_epochs: Total epochs
        metrics: Epoch metrics
    """
    logger.info(f"Epoch {epoch}/{total_epochs}")
    log_metrics(logger, metrics, prefix="  ")


def log_config(logger: logging.Logger, config: dict):
    """
    Log configuration dictionary.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    task: str = "Processing"
):
    """
    Log progress information.
    
    Args:
        logger: Logger instance
        current: Current item
        total: Total items
        task: Task description
    """
    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"{task}: {current}/{total} ({percentage:.1f}%)")


class ProgressLogger:
    """
    Context manager for logging progress.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        task: str = "Processing",
        log_every: int = 100
    ):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance
            total: Total items to process
            task: Task description
            log_every: Log progress every N items
        """
        self.logger = logger
        self.total = total
        self.task = task
        self.log_every = log_every
        self.current = 0
    
    def __enter__(self):
        """Enter context."""
        self.logger.info(f"Starting {self.task}: {self.total} items")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is None:
            self.logger.info(f"Completed {self.task}: {self.current}/{self.total}")
        else:
            self.logger.error(f"Failed {self.task}: {exc_val}")
    
    def update(self, n: int = 1):
        """
        Update progress.
        
        Args:
            n: Number of items processed
        """
        self.current += n
        
        if self.current % self.log_every == 0:
            log_progress(self.logger, self.current, self.total, self.task)
