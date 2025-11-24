Outputs:
- validation_results.xlsx / .csv: Validation metrikleri (tüm modeller)
- best_model_test_metrics.csv: En iyi modelin test sonuclari
- best_model_confusion_matrix.png, best_model_roc_curves.png
- pca_explained_variance.png, pca_scatter_2d.png, lda_scatter_2d.png
- shap_* : SHAP gorselleri
- boxplot_*.png, pearson_correlation_heatmap.png

Notlar:
- LDA n_components otomatik olarak sinif sayisina gore ayarlandi 
- KernelExplainer agirdir; biraz zaman alabiliyor.
