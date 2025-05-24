import os
import sys
from radar.visualization import ExperimentVisualizer

def main():
    """
    使用方法:
    python visualize_results.py <results_directory> [output_directory]
    
    参数:
        results_directory: 包含returns.json的实验结果目录
        output_directory: (可选) 保存可视化结果的目录
    """
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("请提供实验结果目录路径！")
        print("使用方法: python visualize_results.py <results_directory> [output_directory]")
        sys.exit(1)
        
    # 获取目录路径
    results_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 验证输入目录是否存在
    if not os.path.exists(results_dir):
        print(f"错误：目录 {results_dir} 不存在！")
        sys.exit(1)
        
    try:
        # 创建可视化器
        visualizer = ExperimentVisualizer(results_dir)
        
        # 如果指定了输出目录，创建完整报告
        if output_dir:
            print(f"正在生成完整可视化报告到 {output_dir}...")
            summary = visualizer.generate_summary_report(output_dir)
            print("\n数值统计摘要:")
            for category, stats in summary.items():
                print(f"\n{category}:")
                for metric, value in stats.items():
                    print(f"  {metric}: {value:.4f}")
        else:
            # 否则，只显示图表
            print("显示训练曲线...")
            visualizer.plot_training_curves()
            
            print("显示域统计信息...")
            visualizer.plot_domain_statistics()
            
            print("显示回报分布...")
            visualizer.plot_returns_distribution()
            
            print("显示箱线图比较...")
            visualizer.plot_comparison_boxplot()
            
        print("\n可视化完成！")
        
    except Exception as e:
        print(f"错误：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 