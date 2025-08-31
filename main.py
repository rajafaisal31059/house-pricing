#!/usr/bin/env python3

import os
import sys
import importlib.util
import subprocess
from pathlib import Path

class AlgorithmRunner:
    def __init__(self):
        self.algorithms_dir = Path("algorithms")
        self.available_algorithms = self._discover_algorithms()
        
    def _discover_algorithms(self):
       
        algorithms = {}
        if self.algorithms_dir.exists():
            for file_path in self.algorithms_dir.glob("*.py"):
                if file_path.name != "__init__.py":
                    # Extract algorithm name from filename
                    name = file_path.stem.replace("_", " ").title()
                    algorithms[file_path.name] = {
                        "name": name,
                        "path": file_path,
                        "full_path": str(file_path.absolute())
                    }
        return algorithms
    
    def display_menu(self):
        
        print(f"\n")
        print("  MACHINE LEARNING ALGORITHM RUNNER  ")
        print(f"\n")
      

        
        for i, (filename, info) in enumerate(self.available_algorithms.items(), 1):
            print(f"{i:2d}. {info['name']} ({filename})")
        
        print(f"{len(self.available_algorithms) + 1:2d}. Run All Algorithms")
        print(f"{len(self.available_algorithms) + 2:2d}. View Dataset Information")
        print(f"{len(self.available_algorithms) + 3:2d}. Exit")
    
    def get_user_choice(self):
       
        while True:
            try:
                choice = input("\nSelect an option (1-{}): ".format(
                    len(self.available_algorithms) + 3
                )).strip()
                
                if choice.lower() in ['q', 'quit', 'exit']:
                    return len(self.available_algorithms) + 3
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.available_algorithms) + 3:
                    return choice_num
                else:
                    print("Invalid choice. Please enter a number between 1 and {}.".format(
                        len(self.available_algorithms) + 3
                    ))
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
    
    def run_algorithm(self, algorithm_file):
       
        try:
            print(f"\nRunning {algorithm_file['name']}...")
            print(f"File: {algorithm_file['filename']}")
            print("-" * 50)
            
            
            result = subprocess.run(
                [sys.executable, algorithm_file['path']],
                capture_output=False,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                print(f"\n{algorithm_file['name']} completed successfully!")
            else:
                print(f"\n {algorithm_file['name']} encountered an error.")
                
        except Exception as e:
            print(f"\nError running {algorithm_file['name']}: {str(e)}")
    
    def run_all_algorithms(self):
        print("\nRunning all algorithms sequentially...")
        
        for filename, info in self.available_algorithms.items():
            print(f"\n{'='*20} {info['name']} {'='*20}")
            self.run_algorithm({
                'name': info['name'],
                'filename': filename,
                'path': info['path']
            })
            print(f"{'='*20} Completed {'='*20}")
            
            if filename != list(self.available_algorithms.keys())[-1]:  
                continue_choice = input("\nContinue to next algorithm? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', '']:
                    print("⏹Stopping execution.")
                    break
    
    def view_dataset_info(self):

        print("\n DATASET INFORMATION")
        datasets = [
            ("Boston Housing", "data/boston.csv", "MEDV", "Housing prices in Boston suburbs"),
            ("California Housing", "data/california.csv", "median_house_value", "California housing prices"),
            ("New York Airbnb", "data/newyork.csv", "price", "Airbnb listings in New York")
        ]
        
        for name, path, target, description in datasets:
            file_path = Path(path)
            if file_path.exists():
                size = file_path.stat().st_size / 1024  # Size in KB
                print(f" {name}")
                print(f"   Path: {path}")
                print(f"   Target: {target}")
                print(f"   Description: {description}")
                print(f"   Size: {size:.1f} KB")
                print()
            else:
                print(f"{name}: File not found at {path}")
                print()
    
    def run(self):
       
        if not self.available_algorithms:
            print("No algorithm files found in the algorithms directory.")
            print("Please ensure you have algorithm files in the algorithms/ folder.")
            return
        
        while True:
            self.display_menu()
            choice = self.get_user_choice()
            
            if choice == len(self.available_algorithms) + 3:  
                print("\nThanks for using the Algorithm Runner")
                break
            elif choice == len(self.available_algorithms) + 2:  
                self.view_dataset_info()
                input("\nPress Enter to continue...")
            elif choice == len(self.available_algorithms) + 1:  
                self.run_all_algorithms()
                input("\nPress Enter to continue...")
            else:  
                algorithm_files = list(self.available_algorithms.values())
                selected_algorithm = algorithm_files[choice - 1]
                
                self.run_algorithm({
                    'name': selected_algorithm['name'],
                    'filename': selected_algorithm['path'].name,
                    'path': selected_algorithm['path']
                })
                
                input("\nPress Enter to continue...")

def main():
    
    
    if not Path("algorithms").exists():
        print("Error: 'algorithms' directory not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    
    if not Path("data").exists():
        print("Warning: 'data' directory not found!")
        print("Some algorithms may not work without the required datasets.")
    
    runner = AlgorithmRunner()
    runner.run()

if __name__ == "__main__":
    main()
