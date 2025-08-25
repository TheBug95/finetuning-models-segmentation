#!/usr/bin/env python3
"""
Script de comparación entre SAM2, MedSAM2 y MobileSAM2 con diferentes métodos de fine-tuning.
Este script ejecuta entrenamientos comparativos y genera reportes de rendimiento.
"""

import os
import json
import subprocess
import time
from pathlib import Path

# Configuración de experimentos
MODELS = ["sam2", "medsam2", "mobilesam2"]
METHODS = ["baseline", "lora", "qlora"]
DATASETS = ["cataract", "retinopathy"]

def run_finetune(model, method, dataset, dataset_root, epochs=3, batch_size=2):
    """Ejecutar un experimento de fine-tuning."""
    print(f"\n🚀 Ejecutando: {model} + {method} en {dataset}")
    print("="*60)
    
    cmd = [
        "python", "finetune.py",
        "--model", model,
        "--method", method,
        "--dataset", dataset,
        "--dataset-root", dataset_root,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", "1e-4"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hora timeout
        end_time = time.time()
        
        success = result.returncode == 0
        execution_time = end_time - start_time
        
        return {
            "model": model,
            "method": method,
            "dataset": dataset,
            "success": success,
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    except subprocess.TimeoutExpired:
        return {
            "model": model,
            "method": method,
            "dataset": dataset,
            "success": False,
            "execution_time": 3600,
            "error": "Timeout expired",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    except Exception as e:
        return {
            "model": model,
            "method": method,
            "dataset": dataset,
            "success": False,
            "execution_time": 0,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def load_training_metrics(model, method, dataset, epochs):
    """Cargar métricas de entrenamiento desde el archivo JSON."""
    model_dir = f"finetuned-{model}-{method}-{dataset}-epochs{epochs}"
    metrics_file = os.path.join(model_dir, "training_metrics.json")
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def generate_comparison_report(results, output_file="comparison_report.md"):
    """Generar reporte de comparación en Markdown."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Comparación: SAM2, MedSAM2 y MobileSAM2\n\n")
        f.write(f"Generado el: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Resumen ejecutivo
        f.write("## 📊 Resumen Ejecutivo\n\n")
        successful_runs = sum(1 for r in results if r['success'])
        total_runs = len(results)
        f.write(f"- **Experimentos totales**: {total_runs}\n")
        f.write(f"- **Experimentos exitosos**: {successful_runs}\n")
        f.write(f"- **Tasa de éxito**: {successful_runs/total_runs*100:.1f}%\n\n")
        
        # Tabla de resultados por modelo
        f.write("## 🤖 Resultados por Modelo\n\n")
        f.write("| Modelo | Método | Dataset | Estado | Tiempo (min) | Loss Final |\n")
        f.write("|--------|--------|---------|--------|--------------|------------|\n")
        
        for result in results:
            if result['success']:
                # Intentar cargar métricas
                metrics = load_training_metrics(
                    result['model'], 
                    result['method'], 
                    result['dataset'], 
                    3  # epochs por defecto
                )
                final_loss = f"{metrics['final_loss']:.6f}" if metrics else "N/A"
            else:
                final_loss = "❌ Error"
            
            status = "✅ Exitoso" if result['success'] else "❌ Falló"
            time_min = f"{result['execution_time']/60:.1f}"
            
            f.write(f"| {result['model']} | {result['method']} | {result['dataset']} | {status} | {time_min} | {final_loss} |\n")
        
        # Análisis por método
        f.write("\n## ⚙️ Análisis por Método de Fine-tuning\n\n")
        
        method_stats = {}
        for method in METHODS:
            method_results = [r for r in results if r['method'] == method and r['success']]
            if method_results:
                avg_time = sum(r['execution_time'] for r in method_results) / len(method_results)
                method_stats[method] = {
                    'count': len(method_results),
                    'avg_time': avg_time,
                    'success_rate': len(method_results) / sum(1 for r in results if r['method'] == method) * 100
                }
        
        for method, stats in method_stats.items():
            f.write(f"### {method.upper()}\n")
            f.write(f"- **Experimentos exitosos**: {stats['count']}\n")
            f.write(f"- **Tiempo promedio**: {stats['avg_time']/60:.1f} minutos\n")
            f.write(f"- **Tasa de éxito**: {stats['success_rate']:.1f}%\n\n")
        
        # Recomendaciones
        f.write("## 💡 Recomendaciones\n\n")
        f.write("### Para Fine-tuning Básico (Baseline)\n")
        f.write("- ✅ **Pros**: Máxima flexibilidad de entrenamiento\n")
        f.write("- ⚠️ **Contras**: Mayor uso de memoria y tiempo de entrenamiento\n")
        f.write("- 🎯 **Recomendado para**: Cuando se tiene suficiente memoria GPU y se busca máximo rendimiento\n\n")
        
        f.write("### Para LoRA\n")
        f.write("- ✅ **Pros**: Balance entre eficiencia y rendimiento\n")
        f.write("- ✅ **Pros**: Menor uso de memoria que baseline\n")
        f.write("- 🎯 **Recomendado para**: Mayoría de casos de uso prácticos\n\n")
        
        f.write("### Para QLoRA\n")
        f.write("- ✅ **Pros**: Mínimo uso de memoria\n")
        f.write("- ⚠️ **Contras**: Posible pequeña pérdida de rendimiento\n")
        f.write("- 🎯 **Recomendado para**: GPUs con memoria limitada\n\n")
        
        # Errores encontrados
        failed_results = [r for r in results if not r['success']]
        if failed_results:
            f.write("## ❌ Errores Encontrados\n\n")
            for result in failed_results:
                f.write(f"### {result['model']} + {result['method']} en {result['dataset']}\n")
                if 'error' in result:
                    f.write(f"```\n{result['error']}\n```\n\n")
                elif result.get('stderr'):
                    f.write(f"```\n{result['stderr'][:500]}...\n```\n\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comparación de modelos SAM2, MedSAM2 y MobileSAM2")
    parser.add_argument("--dataset-root", required=True,
                       help="Directorio raíz del dataset")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                       help="Modelos a comparar")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS,
                       help="Métodos a comparar")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS,
                       help="Datasets a usar")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Número de épocas por experimento")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Tamaño de batch")
    parser.add_argument("--dry-run", action="store_true",
                       help="Solo mostrar qué experimentos se ejecutarían")
    
    args = parser.parse_args()
    
    # Verificar que existe el dataset
    if not os.path.exists(args.dataset_root):
        print(f"❌ Error: Dataset root no existe: {args.dataset_root}")
        return 1
    
    # Generar lista de experimentos
    experiments = []
    for model in args.models:
        for method in args.methods:
            for dataset in args.datasets:
                experiments.append((model, method, dataset))
    
    print(f"📋 Se ejecutarán {len(experiments)} experimentos:")
    for i, (model, method, dataset) in enumerate(experiments, 1):
        print(f"  {i:2d}. {model} + {method} en {dataset}")
    
    if args.dry_run:
        print("\n🏃‍♂️ Dry run activado - no se ejecutarán experimentos")
        return 0
    
    print(f"\n⏱️  Tiempo estimado: {len(experiments) * 10:.0f} minutos")
    input("Presiona Enter para continuar o Ctrl+C para cancelar...")
    
    # Ejecutar experimentos
    results = []
    total_start_time = time.time()
    
    for i, (model, method, dataset) in enumerate(experiments, 1):
        print(f"\n📊 Progreso: {i}/{len(experiments)}")
        
        result = run_finetune(
            model, method, dataset, args.dataset_root,
            epochs=args.epochs, batch_size=args.batch_size
        )
        results.append(result)
        
        if result['success']:
            print(f"✅ Completado en {result['execution_time']/60:.1f} minutos")
        else:
            print(f"❌ Falló: {result.get('error', 'Error desconocido')}")
    
    total_time = time.time() - total_start_time
    
    # Generar reporte
    print(f"\n📝 Generando reporte de comparación...")
    generate_comparison_report(results)
    
    # Guardar resultados raw
    with open("comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 COMPARACIÓN COMPLETADA")
    print(f"⏱️  Tiempo total: {total_time/60:.1f} minutos")
    print(f"📄 Reporte generado: comparison_report.md")
    print(f"📁 Resultados raw: comparison_results.json")
    
    # Resumen rápido
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Experimentos exitosos: {successful}/{len(results)}")

if __name__ == "__main__":
    exit(main())
