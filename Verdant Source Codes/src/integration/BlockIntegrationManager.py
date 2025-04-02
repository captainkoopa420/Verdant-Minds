import numpy as np
import time
from typing import Dict, List, Any, Optional

class BlockIntegrationManager:
    """
    Manages interactions and information flow between processing blocks 
    in the Unified Synthetic Mind system.
    
    Responsibilities:
    - Track information transfer between blocks
    - Analyze cross-block communication patterns
    - Monitor block performance and interaction efficiency
    - Detect potential integration bottlenecks
    """
    
    def __init__(self, system):
        """
        Initialize the Block Integration Manager.
        
        Args:
            system: The Unified Synthetic Mind system
        """
        self.system = system
        
        # Detailed interaction tracking
        self.interaction_logs = {}
        
        # Performance metrics for each block
        self.block_performance = {}
        
        # Cross-block communication patterns
        self.communication_patterns = {}
        
        # Configuration for tracking and analysis
        self.config = {
            "max_log_entries": 1000,
            "performance_window": 100,  # Number of interactions to track for performance
            "information_transfer_threshold": 0.5  # Minimum significant information transfer
        }
    
    def analyze_block_interactions(self, chunk):
        """
        Analyze interactions between blocks during chunk processing.
        
        Args:
            chunk: Cognitive chunk being processed
        
        Returns:
            Dictionary of interaction details
        """
        interaction_data = {}
        current_time = time.time()
        
        # Iterate through all blocks that processed the chunk
        for section_name, section_content in chunk.sections.items():
            # Extract block name from section name
            block_name = self._extract_block_name(section_name)
            
            if block_name:
                # Track block interaction details
                interaction_data[block_name] = {
                    "timestamp": current_time,
                    "section_content": self._summarize_section(section_content),
                    "information_transfer": self._calculate_information_transfer(section_content)
                }
                
                # Update block performance tracking
                self._update_block_performance(block_name, section_content)
                
                # Track communication patterns
                self._track_communication_patterns(block_name, section_content)
        
        return interaction_data
    
    def _extract_block_name(self, section_name: str) -> Optional[str]:
        """
        Extract block name from section name.
        
        Args:
            section_name: Name of the section (e.g., 'reasoning_section')
        
        Returns:
            Extracted block name or None
        """
        block_mapping = {
            "sensory_input_section": "SensoryInput",
            "pattern_recognition_section": "PatternRecognition",
            "memory_section": "MemoryStorage",
            "internal_communication_section": "InternalCommunication",
            "reasoning_section": "ReasoningPlanning",
            "ethics_king_section": "EthicsValues",
            "action_selection_section": "ActionSelection",
            "language_processing_section": "LanguageProcessing",
            "continual_learning_section": "ContinualLearning"
        }
        
        return block_mapping.get(section_name)
    
    def _summarize_section(self, section_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of section content for lightweight tracking.
        
        Args:
            section_content: Content of a processing block section
        
        Returns:
            Summarized section information
        """
        summary = {
            "keys": list(section_content.keys()),
            "total_size": len(str(section_content)),
            "main_metrics": {}
        }
        
        # Extract some common metrics across different block types
        metric_extractors = {
            "confidence_score": lambda d: d.get("confidence_score", None),
            "inference_count": lambda d: len(d.get("inferences", {}).get("deductive", [])) if isinstance(d.get("inferences"), dict) else None,
            "ethical_status": lambda d: d.get("evaluation", {}).get("status", None),
            "concepts_activated": lambda d: len(d.get("activated_concepts", [])) if isinstance(d.get("activated_concepts"), list) else None
        }
        
        for metric_name, extractor in metric_extractors.items():
            metric_value = extractor(section_content)
            if metric_value is not None:
                summary["main_metrics"][metric_name] = metric_value
        
        return summary
    
    def _calculate_information_transfer(self, section_content: Dict[str, Any]) -> float:
        """
        Calculate the amount of meaningful information transferred by a block.
        
        Args:
            section_content: Content of a processing block section
        
        Returns:
            Information transfer score (0-1)
        """
        # Base calculation based on various indicators of meaningful processing
        transfer_indicators = [
            len(section_content.get("inferences", {})),
            len(section_content.get("activated_concepts", [])),
            section_content.get("confidence_score", 0),
            1 if section_content.get("evaluation", {}).get("status") == "review_needed" else 0
        ]
        
        # Calculate transfer as normalized average of indicators
        transfer_score = np.mean(transfer_indicators) / len(transfer_indicators)
        
        return max(0, min(1, transfer_score))
    
    def _update_block_performance(self, block_name: str, section_content: Dict[str, Any]):
        """
        Update performance tracking for a specific block.
        
        Args:
            block_name: Name of the processing block
            section_content: Content of the block's processing section
        """
        # Initialize performance tracking for the block if not exists
        if block_name not in self.block_performance:
            self.block_performance[block_name] = {
                "total_processing_count": 0,
                "information_transfer_history": [],
                "avg_information_transfer": 0.0,
                "performance_metrics": {}
            }
        
        # Update performance metrics
        block_perf = self.block_performance[block_name]
        block_perf["total_processing_count"] += 1
        
        # Calculate information transfer
        transfer_score = self._calculate_information_transfer(section_content)
        block_perf["information_transfer_history"].append(transfer_score)
        
        # Maintain a fixed-size history
        if len(block_perf["information_transfer_history"]) > self.config["performance_window"]:
            block_perf["information_transfer_history"].pop(0)
        
        # Update average information transfer
        block_perf["avg_information_transfer"] = np.mean(block_perf["information_transfer_history"])
    
    def _track_communication_patterns(self, block_name: str, section_content: Dict[str, Any]):
        """
        Track communication patterns between blocks.
        
        Args:
            block_name: Name of the processing block
            section_content: Content of the block's processing section
        """
        # Check for cross-block insights or communication markers
        cross_block_insights = section_content.get("cross_block_insights", [])
        
        for insight in cross_block_insights:
            source_blocks = insight.get("source_blocks", [])
            target_blocks = insight.get("target_blocks", [])
            
            # Create or update communication pattern keys
            for source in source_blocks:
                for target in target_blocks:
                    key = f"{source}->{target}"
                    
                    if key not in self.communication_patterns:
                        self.communication_patterns[key] = {
                            "interaction_count": 0,
                            "information_transfer_history": [],
                            "avg_transfer_strength": 0.0
                        }
                    
                    pattern = self.communication_patterns[key]
                    pattern["interaction_count"] += 1
                    
                    # Calculate transfer strength
                    transfer_strength = insight.get("importance", 0.5)
                    pattern["information_transfer_history"].append(transfer_strength)
                    
                    # Maintain fixed-size history
                    if len(pattern["information_transfer_history"]) > self.config["performance_window"]:
                        pattern["information_transfer_history"].pop(0)
                    
                    # Update average transfer strength
                    pattern["avg_transfer_strength"] = np.mean(
                        pattern["information_transfer_history"]
                    )
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary containing performance metrics and insights
        """
        report = {
            "overall_performance": {
                "total_interactions": sum(
                    block_data["total_processing_count"]
                    for block_data in self.block_performance.values()
                ),
                "avg_system_information_transfer": np.mean([
                    block_data["avg_information_transfer"]
                    for block_data in self.block_performance.values()
                ])
            },
            "block_performance": {
                block_name: {
                    "total_processing_count": block_data["total_processing_count"],
                    "avg_information_transfer": block_data["avg_information_transfer"]
                }
                for block_name, block_data in self.block_performance.items()
            },
            "communication_patterns": {
                pattern_key: {
                    "interaction_count": pattern_data["interaction_count"],
                    "avg_transfer_strength": pattern_data["avg_transfer_strength"]
                }
                for pattern_key, pattern_data in self.communication_patterns.items()
            }
        }
        
        # Identify potential integration bottlenecks
        report["integration_bottlenecks"] = self._detect_integration_bottlenecks()
        
        return report
    
    def _detect_integration_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Detect potential integration bottlenecks in block interactions.
        
        Returns:
            List of detected bottleneck information
        """
        bottlenecks = []
        
        # Detect low-performing blocks
        for block_name, block_data in self.block_performance.items():
            if block_data["avg_information_transfer"] < self.config["information_transfer_threshold"]:
                bottlenecks.append({
                    "block": block_name,
                    "avg_information_transfer": block_data["avg_information_transfer"],
                    "total_processing_count": block_data["total_processing_count"],
                    "recommendation": "Review block implementation for potential optimization"
                })
        
        # Detect weak communication patterns
        for pattern_key, pattern_data in self.communication_patterns.items():
            if pattern_data["avg_transfer_strength"] < self.config["information_transfer_threshold"]:
                source, target = pattern_key.split('->')
                bottlenecks.append({
                    "source_block": source,
                    "target_block": target,
                    "avg_transfer_strength": pattern_data["avg_transfer_strength"],
                    "interaction_count": pattern_data["interaction_count"],
                    "recommendation": "Investigate and improve cross-block communication"
                })
        
        return bottlenecks
    
    def reset_tracking(self):
        """
        Reset all tracking data to start fresh.
        """
        self.interaction_logs.clear()
        self.block_performance.clear()
        self.communication_patterns.clear()