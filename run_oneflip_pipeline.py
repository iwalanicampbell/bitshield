#!/usr/bin/env python3
"""
Complete Pipeline: OneFlip Quantized Model Injection → BitShield Defense Analysis
==================================================================================

This script orchestrates the end-to-end workflow:
1. Integrate OneFlip quantized model with BitShield
2. Compile backdoored model to native binary
3. Simulate bit-flip attacks
4. Analyze defense effectiveness

Usage:
    python run_oneflip_pipeline.py \\
        -oneflip_model /path/to/model.pth \\
        -bitshield_dir /path/to/bitshield \\
        -dataset CIFAR10 \\
        -num_flips 1000
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from oneflip_adapter import integrate_oneflip_to_bitshield


class OneFlipBitShieldPipeline:
    """Orchestrate OneFlip ↔ BitShield integration and analysis"""
    
    def __init__(self, bitshield_dir: str, output_dir: Optional[str] = None):
        """
        Initialize pipeline.
        
        Args:
            bitshield_dir: Path to BitShield project
            output_dir: Where to save results (default: bitshield/results/oneflip_{timestamp})
        """
        self.bitshield_dir = Path(bitshield_dir)
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = self.bitshield_dir / 'results' / f'oneflip_{timestamp}'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / 'pipeline.log'
        
    def log(self, message: str, level: str = 'INFO'):
        """Write to log file and stdout."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def step(self, num: int, total: int, title: str):
        """Print step header."""
        header = f"\n{'='*70}"
        header += f"\nSTEP {num}/{total}: {title}"
        header += f"\n{'='*70}\n"
        self.log(header)
        print(header, end='')
    
    def step_integrate(self, oneflip_model: str, dataset: str, arch: str) -> Dict:
        """
        STEP 1: Integrate OneFlip model with BitShield
        """
        self.step(1, 4, "Integrate OneFlip Model")
        
        self.log(f"Model: {oneflip_model}")
        self.log(f"Dataset: {dataset}, Architecture: {arch}")
        
        result = integrate_oneflip_to_bitshield(
            oneflip_model,
            str(self.bitshield_dir),
            dataset=dataset,
            model_arch=arch
        )
        
        # Save integration metadata
        integration_file = self.output_dir / 'integration_metadata.json'
        with open(integration_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        self.log(f"✓ Integration complete")
        self.log(f"  ONNX: {result['onnx_path']}")
        self.log(f"  Config: {result['config_path']}")
        
        return result
    
    def step_compile_binary(self, config_path: str, model_name: str) -> str:
        """
        STEP 2: Compile ONNX model to native binary (.so)
        
        Uses BitShield's buildmodels.py to compile the quantized model.
        """
        self.step(2, 4, "Compile to Binary")
        
        self.log(f"Config: {config_path}")
        self.log(f"Model: {model_name}")
        
        # Extract model info from config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        dataset = config['dataset']
        arch = config['architecture']
        
        self.log(f"Running: buildmodels.py for {arch} on {dataset}")
        
        # Call BitShield's build process
        build_script = self.bitshield_dir / 'buildmodels.py'
        if not build_script.exists():
            self.log("⚠ buildmodels.py not found, skipping binary compilation", 'WARN')
            return None
        
        try:
            cmd = [
                'python', str(build_script),
                '-model', model_name,
                '-dataset', dataset
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.bitshield_dir),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                self.log(f"✗ Build failed with return code {result.returncode}", 'ERROR')
                self.log(f"stdout: {result.stdout}", 'ERROR')
                self.log(f"stderr: {result.stderr}", 'ERROR')
                return None
            
            self.log(f"✓ Binary compiled successfully")
            
            # Find the compiled binary
            built_dir = self.bitshield_dir / 'built'
            so_file = built_dir / f"{arch}_{dataset}_quantized.so"
            
            if so_file.exists():
                self.log(f"  Binary: {so_file}")
                return str(so_file)
            else:
                self.log(f"⚠ Could not find expected binary", 'WARN')
                return None
            
        except subprocess.TimeoutExpired:
            self.log("✗ Build timeout (>1 hour)", 'ERROR')
            return None
        except Exception as e:
            self.log(f"✗ Build failed: {e}", 'ERROR')
            return None
    
    def step_attack_simulation(self, model_name: str, dataset: str, 
                               num_flips: int, num_trials: int) -> Optional[str]:
        """
        STEP 3: Run bit-flip attack simulations
        
        Uses BitShield's attacksim.py to test defense effectiveness.
        """
        self.step(3, 4, "Attack Simulation")
        
        self.log(f"Model: {model_name}")
        self.log(f"Dataset: {dataset}")
        self.log(f"Bit flips: {num_flips}, Trials: {num_trials}")
        
        attacksim_script = self.bitshield_dir / 'attacksim.py'
        if not attacksim_script.exists():
            self.log("⚠ attacksim.py not found, skipping attack simulation", 'WARN')
            return None
        
        try:
            cmd = [
                'python', str(attacksim_script),
                '-model', model_name,
                '-dataset', dataset,
                '-num_flips', str(num_flips),
                '-num_trials', str(num_trials),
                '-output', str(self.output_dir)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.bitshield_dir),
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode != 0:
                self.log(f"⚠ Attack simulation returned code {result.returncode}", 'WARN')
                self.log(f"stdout: {result.stdout[-500:]}", 'WARN')  # Last 500 chars
                # Don't fail on this - results may still be partial
            else:
                self.log(f"✓ Attack simulation complete")
            
            # Find results
            results_file = self.output_dir / 'attack_results.pkl'
            if results_file.exists():
                self.log(f"  Results: {results_file}")
                return str(results_file)
            else:
                self.log(f"⚠ Results file not found", 'WARN')
                return None
            
        except subprocess.TimeoutExpired:
            self.log("✗ Attack simulation timeout (>2 hours)", 'ERROR')
            return None
        except Exception as e:
            self.log(f"✗ Attack simulation failed: {e}", 'ERROR')
            return None
    
    def step_analysis(self, attack_results: Optional[str]) -> Dict:
        """
        STEP 4: Analyze results and generate report
        """
        self.step(4, 4, "Analyze Results")
        
        analysis = {
            'status': 'complete',
            'timestamp': datetime.now().isoformat(),
            'output_dir': str(self.output_dir),
        }
        
        if attack_results and os.path.exists(attack_results):
            self.log(f"Analyzing: {attack_results}")
            
            # TODO: Parse attack_results and compute statistics
            analysis['attack_results_file'] = attack_results
            analysis['summary'] = {
                'detection_rate': 'N/A (requires parsing)',
                'avg_impact': 'N/A (requires parsing)',
                'defense_effectiveness': 'N/A (requires parsing)'
            }
        else:
            self.log("⚠ No attack results to analyze", 'WARN')
            analysis['status'] = 'partial'
        
        # Save analysis
        analysis_file = self.output_dir / 'analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.log(f"✓ Analysis complete")
        self.log(f"  Report: {analysis_file}")
        
        return analysis
    
    def run(self, oneflip_model: str, dataset: str, arch: str,
            num_flips: int = 1000, num_trials: int = 10) -> Dict:
        """
        Execute complete pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        self.log("="*70)
        self.log("OneFlip Quantized Model → BitShield Defense Analysis Pipeline")
        self.log("="*70)
        self.log(f"Output: {self.output_dir}\n")
        
        try:
            # Step 1: Integration
            integration = self.step_integrate(oneflip_model, dataset, arch)
            config_path = integration['config_path']
            model_name = f"Q{arch}_{dataset}"
            
            # Step 2: Compilation
            binary_path = self.step_compile_binary(config_path, model_name)
            
            # Step 3: Attack Simulation
            attack_results = self.step_attack_simulation(
                model_name, dataset, num_flips, num_trials
            )
            
            # Step 4: Analysis
            analysis = self.step_analysis(attack_results)
            
            # Final summary
            self.log("\n" + "="*70)
            self.log("PIPELINE COMPLETE")
            self.log("="*70)
            self.log(f"Output directory: {self.output_dir}")
            self.log(f"Log file: {self.log_file}")
            
            return {
                'status': 'success',
                'output_dir': str(self.output_dir),
                'integration': integration,
                'binary': binary_path,
                'attack_results': attack_results,
                'analysis': analysis
            }
            
        except Exception as e:
            self.log(f"\n✗ PIPELINE FAILED: {e}", 'ERROR')
            return {
                'status': 'failed',
                'error': str(e),
                'output_dir': str(self.output_dir)
            }


def main():
    """Command-line entry point."""
    
    parser = argparse.ArgumentParser(
        description='OneFlip Quantized Model → BitShield Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_oneflip_pipeline.py \\
    -oneflip_model saved_model/resnet_CIFAR10/clean_model_1_int8_state.pth \\
    -bitshield_dir e:\\GithubReps\\bitshield

  # With custom parameters
  python run_oneflip_pipeline.py \\
    -oneflip_model saved_model/resnet_CIFAR10/clean_model_1_int8_state.pth \\
    -bitshield_dir e:\\GithubReps\\bitshield \\
    -dataset CIFAR10 \\
    -arch resnet \\
    -num_flips 5000 \\
    -num_trials 20
        """
    )
    
    parser.add_argument(
        '-oneflip_model', required=True,
        help='Path to OneFlip quantized model checkpoint'
    )
    parser.add_argument(
        '-bitshield_dir', required=True,
        help='BitShield project directory'
    )
    parser.add_argument(
        '-dataset', default='CIFAR10',
        help='Dataset (CIFAR10, CIFAR100, ImageNet, etc.)'
    )
    parser.add_argument(
        '-arch', default='resnet',
        help='Model architecture (resnet, preactres, vgg, etc.)'
    )
    parser.add_argument(
        '-num_flips', type=int, default=1000,
        help='Number of bit-flip simulations'
    )
    parser.add_argument(
        '-num_trials', type=int, default=10,
        help='Number of trials per configuration'
    )
    parser.add_argument(
        '-output_dir',
        help='Custom output directory (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.oneflip_model):
        print(f"ERROR: OneFlip model not found: {args.oneflip_model}")
        sys.exit(1)
    
    if not os.path.isdir(args.bitshield_dir):
        print(f"ERROR: BitShield directory not found: {args.bitshield_dir}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = OneFlipBitShieldPipeline(args.bitshield_dir, args.output_dir)
    
    result = pipeline.run(
        oneflip_model=args.oneflip_model,
        dataset=args.dataset,
        arch=args.arch,
        num_flips=args.num_flips,
        num_trials=args.num_trials
    )
    
    sys.exit(0 if result['status'] == 'success' else 1)


if __name__ == '__main__':
    main()
