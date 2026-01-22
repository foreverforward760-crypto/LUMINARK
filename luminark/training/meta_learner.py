"""
Meta-Learning Engine for Recursive Self-Improvement
Learns to improve the learning process itself
"""
import numpy as np
from typing import Dict, List, Optional


class MetaLearningEngine:
    """
    Meta-learner that improves training performance over time
    Learns which hyperparameters work best for different situations
    """
    
    def __init__(self):
        # Track performance history
        self.performance_history = []
        
        # Hyperparameter recommendations
        self.hyperparameter_performance = {}
        
        # Learning rate adaptation
        self.lr_history = []
        self.best_lr = 0.001
        
        # Architecture recommendations
        self.architecture_scores = {}
    
    def record_training_result(self, config: Dict, performance: Dict):
        """
        Record training result for meta-learning
        
        Args:
            config: Dict with training config (lr, batch_size, architecture, etc.)
            performance: Dict with results (final_accuracy, training_time, etc.)
        """
        self.performance_history.append({
            'config': config,
            'performance': performance
        })
        
        # Update hyperparameter performance
        for key, value in config.items():
            if key not in self.hyperparameter_performance:
                self.hyperparameter_performance[key] = {}
            
            value_key = str(value)
            if value_key not in self.hyperparameter_performance[key]:
                self.hyperparameter_performance[key][value_key] = []
            
            self.hyperparameter_performance[key][value_key].append(
                performance.get('final_accuracy', 0.0)
            )
        
        # Update learning rate recommendations
        if 'lr' in config:
            self.lr_history.append((config['lr'], performance.get('final_accuracy', 0.0)))
            self._update_best_lr()
    
    def recommend_hyperparameters(self, context: Dict = None) -> Dict:
        """
        Recommend hyperparameters based on past performance
        
        Args:
            context: Optional context about the task
            
        Returns:
            Dict of recommended hyperparameters
        """
        recommendations = {}
        
        # Recommend learning rate
        recommendations['lr'] = self.best_lr
        
        # Recommend other hyperparameters based on performance
        for param, values in self.hyperparameter_performance.items():
            if param == 'lr':
                continue
            
            # Find best performing value
            best_value = None
            best_avg_performance = -1
            
            for value, performances in values.items():
                avg_perf = np.mean(performances)
                if avg_perf > best_avg_performance:
                    best_avg_performance = avg_perf
                    best_value = value
            
            if best_value is not None:
                # Try to convert back to original type
                try:
                    if '.' in best_value:
                        recommendations[param] = float(best_value)
                    else:
                        recommendations[param] = int(best_value)
                except:
                    recommendations[param] = best_value
        
        return recommendations
    
    def suggest_learning_rate_adjustment(self, current_lr: float, 
                                        recent_performance: List[float]) -> float:
        """
        Suggest learning rate adjustment based on recent performance
        
        Args:
            current_lr: Current learning rate
            recent_performance: List of recent accuracy/loss values
            
        Returns:
            Recommended new learning rate
        """
        if len(recent_performance) < 2:
            return current_lr
        
        # Check if performance is improving
        recent_trend = recent_performance[-1] - recent_performance[-2]
        
        if recent_trend > 0:
            # Improving - can try slightly higher LR
            new_lr = current_lr * 1.05
        elif recent_trend < -0.05:
            # Degrading significantly - reduce LR
            new_lr = current_lr * 0.5
        else:
            # Stable - keep current LR
            new_lr = current_lr
        
        # Bound learning rate
        new_lr = np.clip(new_lr, 1e-6, 1.0)
        
        return float(new_lr)
    
    def analyze_training_patterns(self) -> Dict:
        """
        Analyze patterns in training history
        
        Returns:
            Dict with insights about what works
        """
        if len(self.performance_history) < 5:
            return {
                'insights': ['Not enough data yet for analysis'],
                'best_lr': self.best_lr,
                'total_experiments': len(self.performance_history)
            }
        
        insights = []
        
        # Analyze learning rate impact
        if len(self.lr_history) >= 3:
            lrs, perfs = zip(*self.lr_history[-10:])
            correlation = np.corrcoef(lrs, perfs)[0, 1]
            
            if correlation > 0.5:
                insights.append("Higher learning rates tend to perform better")
            elif correlation < -0.5:
                insights.append("Lower learning rates tend to perform better")
        
        # Analyze performance trends
        recent_perfs = [h['performance'].get('final_accuracy', 0.0) 
                       for h in self.performance_history[-10:]]
        
        if len(recent_perfs) >= 5:
            trend = np.polyfit(range(len(recent_perfs)), recent_perfs, 1)[0]
            
            if trend > 0.01:
                insights.append("Performance is improving over time - keep current approach")
            elif trend < -0.01:
                insights.append("Performance is degrading - consider changing strategy")
        
        return {
            'insights': insights,
            'best_lr': self.best_lr,
            'total_experiments': len(self.performance_history)
        }
    
    def _update_best_lr(self):
        """Update best learning rate based on history"""
        if len(self.lr_history) < 3:
            return
        
        # Find LR with best average performance
        lr_performance = {}
        for lr, perf in self.lr_history:
            if lr not in lr_performance:
                lr_performance[lr] = []
            lr_performance[lr].append(perf)
        
        best_lr = self.best_lr
        best_avg_perf = -1
        
        for lr, perfs in lr_performance.items():
            avg_perf = np.mean(perfs)
            if avg_perf > best_avg_perf:
                best_avg_perf = avg_perf
                best_lr = lr
        
        self.best_lr = best_lr
