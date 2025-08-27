"""
Script de comparación automatizada entre modelos y métodos.
Ejecuta experimentos completos y genera reportes de rendimiento.
"""

import os
import json
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


class BenchmarkRunner:
    """Ejecutor de benchmarks para comparación de modelos."""
    
    def __init__(self, dataset_root: str, output_dir: str = "benchmark_results"):
        """
        Inicializa el ejecutor de benchmarks.
        
        Args:
            dataset_root: Directorio raíz donde están los datasets
            output_dir: Directorio donde guardar resultados
        """
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configuración de experimentos
        self.models = ["sam2", "medsam2", "mobilesam"]
        self.methods = ["baseline", "lora", "qlora"]
        self.datasets = ["cataract", "retinopathy"]
        
        self.results = []
        
    def run_single_experiment(self, 
                            model: str,
                            method: str, 
                            dataset: str,
                            epochs: int = 3,
                            batch_size: int = 2) -> Dict[str, Any]:
        """
        Ejecuta un experimento individual.
        
        Args:
            model: Modelo a entrenar
            method: Método de fine-tuning
            dataset: Dataset a usar
            epochs: Número de épocas
            batch_size: Tamaño del batch
            
        Returns:
            Diccionario con resultados del experimento
        """
        experiment_name = f"{model}_{method}_{dataset}"
        print(f"\n🚀 Ejecutando experimento: {experiment_name}")
        print("="*50)
        
        # Configurar paths
        dataset_path = self.dataset_root / f"{dataset.title()} COCO Segmentation" / "train"
        if not dataset_path.exists():
            dataset_path = self.dataset_root / dataset
        
        if not dataset_path.exists():
            print(f"❌ Dataset no encontrado: {dataset_path}")
            return {
                "model": model,
                "method": method,
                "dataset": dataset,
                "success": False,
                "error": f"Dataset path not found: {dataset_path}"
            }
        
        experiment_output = self.output_dir / experiment_name
        
        # Comando de entrenamiento
        cmd = [
            "python", "train.py",
            "--model", model,
            "--method", method,
            "--dataset", dataset,
            "--dataset-root", str(dataset_path),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--output-dir", str(experiment_output)
        ]
        
        # Ejecutar experimento
        start_time = time.time()
        
        try:
            print(f"Ejecutando: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=7200,  # 2 horas timeout
                cwd=os.getcwd()
            )
            
            execution_time = time.time() - start_time
            success = result.returncode == 0
            
            # Cargar resultados si fue exitoso
            training_summary = {}
            model_info = {}
            
            if success and experiment_output.exists():
                summary_file = experiment_output / "training_summary.json"
                model_info_file = experiment_output / "model_info.json"
                
                if summary_file.exists():
                    with open(summary_file) as f:
                        training_summary = json.load(f)
                        
                if model_info_file.exists():
                    with open(model_info_file) as f:
                        model_info = json.load(f)
            
            result_data = {
                "model": model,
                "method": method,
                "dataset": dataset,
                "success": success,
                "execution_time": execution_time,
                "epochs": epochs,
                "batch_size": batch_size,
                "training_summary": training_summary,
                "model_info": model_info,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if success:
                print(f"✅ Experimento completado en {execution_time:.1f}s")
            else:
                print(f"❌ Experimento falló después de {execution_time:.1f}s")
                print(f"Error: {result.stderr[-500:]}")  # Últimas 500 chars del error
            
            return result_data
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"⏰ Experimento timeout después de {execution_time:.1f}s")
            return {
                "model": model,
                "method": method,
                "dataset": dataset,
                "success": False,
                "execution_time": execution_time,
                "error": "Timeout"
            }
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Error ejecutando experimento: {e}")
            return {
                "model": model,
                "method": method,
                "dataset": dataset,
                "success": False,
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def run_full_benchmark(self, 
                          epochs: int = 3,
                          batch_size: int = 2,
                          models: List[str] = None,
                          methods: List[str] = None,
                          datasets: List[str] = None) -> None:
        """
        Ejecuta el benchmark completo.
        
        Args:
            epochs: Número de épocas por experimento
            batch_size: Tamaño del batch
            models: Lista de modelos (None = todos)
            methods: Lista de métodos (None = todos)
            datasets: Lista de datasets (None = todos)
        """
        models = models or self.models
        methods = methods or self.methods
        datasets = datasets or self.datasets
        
        total_experiments = len(models) * len(methods) * len(datasets)
        
        print(f"🎯 INICIANDO BENCHMARK COMPLETO")
        print("="*60)
        print(f"📊 Experimentos totales: {total_experiments}")
        print(f"🤖 Modelos: {models}")
        print(f"⚙️  Métodos: {methods}")
        print(f"📁 Datasets: {datasets}")
        print(f"📈 Épocas por experimento: {epochs}")
        print("="*60)
        
        experiment_count = 0
        
        for model in models:
            for method in methods:
                for dataset in datasets:
                    experiment_count += 1
                    print(f"\n📈 Progreso: {experiment_count}/{total_experiments}")
                    
                    result = self.run_single_experiment(
                        model, method, dataset, epochs, batch_size
                    )
                    self.results.append(result)
                    
                    # Guardar resultados parciales
                    self.save_results()
        
        print(f"\n🎉 BENCHMARK COMPLETADO")
        print("="*60)
        
        # Generar reporte final
        self.generate_report()
    
    def save_results(self) -> None:
        """Guarda los resultados actuales."""
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self) -> None:
        """Genera reporte de resultados."""
        print("📊 Generando reporte de resultados...")
        
        # Crear DataFrame para análisis
        report_data = []
        
        for result in self.results:
            row = {
                "model": result["model"],
                "method": result["method"],
                "dataset": result["dataset"],
                "success": result["success"],
                "execution_time": result.get("execution_time", 0),
                "epochs": result.get("epochs", 0),
                "batch_size": result.get("batch_size", 0)
            }
            
            # Extraer métricas de entrenamiento si están disponibles
            training_summary = result.get("training_summary", {})
            metrics = training_summary.get("metrics", {})
            
            row.update({
                "final_loss": metrics.get("final_loss"),
                "best_loss": metrics.get("best_loss"),
                "total_training_time": metrics.get("total_time"),
                "improvement": metrics.get("improvement")
            })
            
            # Extraer info del modelo
            model_info = result.get("model_info", {})
            row.update({
                "trainable_parameters": model_info.get("trainable_parameters"),
                "total_parameters": model_info.get("total_parameters"),
                "trainable_percentage": model_info.get("trainable_percentage")
            })
            
            report_data.append(row)
        
        # Crear DataFrame
        df = pd.DataFrame(report_data)
        
        # Guardar CSV
        csv_file = self.output_dir / "benchmark_report.csv"
        df.to_csv(csv_file, index=False)
        
        # Generar estadísticas
        stats = {
            "total_experiments": len(self.results),
            "successful_experiments": sum(1 for r in self.results if r["success"]),
            "failed_experiments": sum(1 for r in self.results if not r["success"]),
            "average_execution_time": df["execution_time"].mean(),
            "total_execution_time": df["execution_time"].sum()
        }
        
        # Estadísticas por categoría
        if not df.empty:
            successful_df = df[df["success"] == True]
            
            if not successful_df.empty:
                stats.update({
                    "best_final_loss_overall": successful_df["final_loss"].min(),
                    "best_model_by_loss": successful_df.loc[successful_df["final_loss"].idxmin()]["model"],
                    "best_method_by_loss": successful_df.loc[successful_df["final_loss"].idxmin()]["method"],
                    "fastest_training": successful_df["execution_time"].min(),
                    "slowest_training": successful_df["execution_time"].max()
                })
                
                # Estadísticas por modelo
                stats["model_performance"] = {}
                for model in successful_df["model"].unique():
                    model_df = successful_df[successful_df["model"] == model]
                    stats["model_performance"][model] = {
                        "avg_final_loss": model_df["final_loss"].mean(),
                        "avg_execution_time": model_df["execution_time"].mean(),
                        "success_rate": len(model_df) / len(df[df["model"] == model])
                    }
                
                # Estadísticas por método
                stats["method_performance"] = {}
                for method in successful_df["method"].unique():
                    method_df = successful_df[successful_df["method"] == method]
                    stats["method_performance"][method] = {
                        "avg_final_loss": method_df["final_loss"].mean(),
                        "avg_execution_time": method_df["execution_time"].mean(),
                        "avg_trainable_percentage": method_df["trainable_percentage"].mean(),
                        "success_rate": len(method_df) / len(df[df["method"] == method])
                    }
        
        # Guardar estadísticas
        stats_file = self.output_dir / "benchmark_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        # Imprimir resumen
        print(f"\n📊 RESUMEN DE RESULTADOS")
        print("="*60)
        print(f"✅ Experimentos exitosos: {stats['successful_experiments']}/{stats['total_experiments']}")
        print(f"⏱️  Tiempo total: {stats['total_execution_time']:.1f}s")
        print(f"📁 Archivos generados:")
        print(f"   - {csv_file}")
        print(f"   - {stats_file}")
        print(f"   - {self.output_dir / 'benchmark_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Ejecutar benchmark de modelos SAM")
    
    parser.add_argument("--dataset-root",
                       required=True,
                       help="Directorio raíz donde están los datasets")
    parser.add_argument("--output-dir",
                       default="benchmark_results",
                       help="Directorio de salida para resultados")
    parser.add_argument("--epochs",
                       type=int,
                       default=3,
                       help="Número de épocas por experimento")
    parser.add_argument("--batch-size",
                       type=int,
                       default=2,
                       help="Tamaño del batch")
    parser.add_argument("--models",
                       nargs="+",
                       choices=["sam2", "medsam2", "mobilesam"],
                       help="Modelos a evaluar (default: todos)")
    parser.add_argument("--methods",
                       nargs="+",
                       choices=["baseline", "lora", "qlora"],
                       help="Métodos a evaluar (default: todos)")
    parser.add_argument("--datasets",
                       nargs="+",
                       choices=["cataract", "retinopathy"],
                       help="Datasets a evaluar (default: todos)")
    
    args = parser.parse_args()
    
    # Verificar que existe el dataset root
    if not os.path.exists(args.dataset_root):
        print(f"❌ Error: Dataset root no existe: {args.dataset_root}")
        return 1
    
    # Crear runner y ejecutar benchmark
    runner = BenchmarkRunner(args.dataset_root, args.output_dir)
    
    runner.run_full_benchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        models=args.models,
        methods=args.methods,
        datasets=args.datasets
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
