"""
Train and save all machine learning models to the models directory.

This script trains all models (Logistic Regression, KNN, SVM, ANN, 
Linear Regression, Polynomial Regression) and saves them as .pkl files
in the models/ directory for later use.

Usage:
    python train_models.py
"""

from model_utils import train_all_models, save_models_to_disk

def main():
    print("=" * 60)
    print("TRAINING CARDIOVASCULAR RISK ASSESSMENT MODELS")
    print("=" * 60)
    print()
    
    # Train all models
    print("Training models...")
    print("-" * 60)
    models = train_all_models(seed=7)
    print("-" * 60)
    
    # Display model metrics
    print("\nModel Performance Summary:")
    print("-" * 60)
    for model_name, model_bundle in models.items():
        metrics = model_bundle.metrics
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    print("\n" + "-" * 60)
    
    # Save models to disk
    print("\nSaving models to disk...")
    print("-" * 60)
    save_models_to_disk(models, models_dir="models")
    print("-" * 60)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nAll models have been trained and saved to the 'models/' directory.")
    print("The Flask API will now load these pre-trained models on startup.")
    print()

if __name__ == "__main__":
    main()
