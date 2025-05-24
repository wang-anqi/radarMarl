import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from os.path import join
import os

class ExperimentVisualizer:
    """实验结果可视化类"""
    
    def __init__(self, results_dir):
        """
        初始化可视化器
        
        参数:
            results_dir: 实验结果目录
        """
        self.results_dir = results_dir
        self.data = self._load_data()
        
        # 设置绘图风格
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def _load_data(self):
        """加载returns.json数据"""
        data_path = join(self.results_dir, 'returns.json')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据文件: {data_path}")
            
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线
        
        包括:
        - 训练回报
        - 测试回报
        - 主角智能体回报
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 训练和测试回报
        episodes = range(len(self.data['training_discounted_returns']))
        test_episodes = range(0, len(episodes), len(episodes)//len(self.data['test_discounted_returns']))
        
        ax1.plot(episodes, self.data['training_discounted_returns'], 
                label='训练折扣回报', alpha=0.7)
        ax1.plot(test_episodes, self.data['test_discounted_returns'], 
                label='测试折扣回报', marker='o')
        ax1.set_xlabel('训练回合')
        ax1.set_ylabel('折扣回报')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('训练过程中的折扣回报变化')
        
        # 主角智能体回报
        ax2.plot(episodes, self.data['protagonist_discounted_returns'], 
                label='主角智能体折扣回报', color='green')
        ax2.set_xlabel('训练回合')
        ax2.set_ylabel('主角智能体回报')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('主角智能体的表现')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_domain_statistics(self, save_path=None):
        """绘制域统计信息"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episodes = range(len(self.data['training_domain_statistic']))
        test_episodes = range(0, len(episodes), 
                            len(episodes)//len(self.data['test_domain_statistic']))
        
        ax.plot(episodes, self.data['training_domain_statistic'], 
                label='训练域统计', alpha=0.7)
        ax.plot(test_episodes, self.data['test_domain_statistic'], 
                label='测试域统计', marker='o')
        
        ax.set_xlabel('训练回合')
        ax.set_ylabel('域统计值')
        ax.legend()
        ax.grid(True)
        ax.set_title('域统计随训练变化')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_returns_distribution(self, save_path=None):
        """绘制回报分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 训练回报分布
        sns.histplot(data=self.data['training_discounted_returns'], 
                    ax=ax1, kde=True)
        ax1.set_title('训练回报分布')
        ax1.set_xlabel('折扣回报')
        ax1.set_ylabel('频次')
        
        # 测试回报分布
        sns.histplot(data=self.data['test_discounted_returns'], 
                    ax=ax2, kde=True, color='orange')
        ax2.set_title('测试回报分布')
        ax2.set_xlabel('折扣回报')
        ax2.set_ylabel('频次')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_comparison_boxplot(self, save_path=None):
        """绘制训练和测试回报的箱线图比较"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_to_plot = [
            self.data['training_discounted_returns'],
            self.data['test_discounted_returns'],
            self.data['protagonist_discounted_returns']
        ]
        
        ax.boxplot(data_to_plot, labels=['训练回报', '测试回报', '主角智能体回报'])
        ax.set_ylabel('折扣回报')
        ax.set_title('不同类型回报的分布比较')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def generate_summary_report(self, save_dir=None):
        """生成完整的可视化报告"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # 绘制所有图表
        self.plot_training_curves(
            join(save_dir, 'training_curves.png') if save_dir else None)
        self.plot_domain_statistics(
            join(save_dir, 'domain_statistics.png') if save_dir else None)
        self.plot_returns_distribution(
            join(save_dir, 'returns_distribution.png') if save_dir else None)
        self.plot_comparison_boxplot(
            join(save_dir, 'comparison_boxplot.png') if save_dir else None)
        
        # 生成数值统计摘要
        summary = {
            '训练统计': {
                '平均训练回报': np.mean(self.data['training_discounted_returns']),
                '训练回报标准差': np.std(self.data['training_discounted_returns']),
                '最大训练回报': np.max(self.data['training_discounted_returns']),
                '最小训练回报': np.min(self.data['training_discounted_returns'])
            },
            '测试统计': {
                '平均测试回报': np.mean(self.data['test_discounted_returns']),
                '测试回报标准差': np.std(self.data['test_discounted_returns']),
                '最大测试回报': np.max(self.data['test_discounted_returns']),
                '最小测试回报': np.min(self.data['test_discounted_returns'])
            },
            '主角智能体统计': {
                '平均回报': np.mean(self.data['protagonist_discounted_returns']),
                '回报标准差': np.std(self.data['protagonist_discounted_returns']),
                '最大回报': np.max(self.data['protagonist_discounted_returns']),
                '最小回报': np.min(self.data['protagonist_discounted_returns'])
            }
        }
        
        if save_dir:
            with open(join(save_dir, 'summary_statistics.json'), 'w') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
        
        return summary 