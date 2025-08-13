"""
Base classes for the carrier analysis pipeline framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Type variable for pipeline data
T = TypeVar('T')
R = TypeVar('R')


@dataclass
class PipelineContext:
    """Context passed through pipeline stages."""
    dataset_type: str
    release: str
    output_path: str
    metadata: Dict[str, Any]
    start_time: datetime
    
    def __init__(self, dataset_type: str, release: str, output_path: str, **kwargs):
        self.dataset_type = dataset_type
        self.release = release
        self.output_path = output_path
        self.metadata = kwargs
        self.start_time = datetime.now()


class PipelineStage(ABC, Generic[T, R]):
    """Abstract base class for pipeline stages."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def process(self, data: T, context: PipelineContext) -> R:
        """
        Process data through this stage.
        
        Args:
            data: Input data from previous stage
            context: Pipeline context with metadata
            
        Returns:
            Processed data for next stage
        """
        pass
    
    async def __call__(self, data: T, context: PipelineContext) -> R:
        """Wrapper to add logging and error handling."""
        try:
            self.logger.info(f"Starting stage: {self.name}")
            result = await self.process(data, context)
            self.logger.info(f"Completed stage: {self.name}")
            return result
        except Exception as e:
            self.logger.error(f"Error in stage {self.name}: {str(e)}")
            raise


class Pipeline:
    """Orchestrates pipeline stages."""
    
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
        self.logger = logging.getLogger(__name__)
    
    async def run(self, initial_data: Any, context: PipelineContext) -> Any:
        """
        Run data through all pipeline stages.
        
        Args:
            initial_data: Initial input data
            context: Pipeline context
            
        Returns:
            Final processed data
        """
        data = initial_data
        
        for stage in self.stages:
            self.logger.info(f"Executing stage: {stage.name}")
            data = await stage(data, context)
            
        return data


class ParallelPipelineStage(PipelineStage[List[T], List[R]]):
    """Base class for stages that can process items in parallel."""
    
    def __init__(self, name: str, max_workers: int = 4):
        super().__init__(name)
        self.max_workers = max_workers
    
    @abstractmethod
    async def process_item(self, item: T, context: PipelineContext) -> R:
        """Process a single item."""
        pass
    
    async def process(self, items: List[T], context: PipelineContext) -> List[R]:
        """Process multiple items in parallel."""
        import asyncio
        
        # Create tasks for all items
        tasks = [self.process_item(item, context) for item in items]
        
        # Process in batches to limit concurrency
        results = []
        for i in range(0, len(tasks), self.max_workers):
            batch = tasks[i:i + self.max_workers]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            
        return results


class ConditionalPipelineStage(PipelineStage[T, T]):
    """Stage that conditionally executes based on context."""
    
    def __init__(self, name: str, condition_fn):
        super().__init__(name)
        self.condition_fn = condition_fn
        self.wrapped_stage: Optional[PipelineStage] = None
    
    def set_wrapped_stage(self, stage: PipelineStage):
        """Set the stage to execute if condition is met."""
        self.wrapped_stage = stage
        return self
    
    async def process(self, data: T, context: PipelineContext) -> T:
        """Process data only if condition is met."""
        if self.condition_fn(context):
            if self.wrapped_stage:
                return await self.wrapped_stage.process(data, context)
        return data
