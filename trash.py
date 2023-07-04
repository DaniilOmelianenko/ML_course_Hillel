## KNeighborsClassifier() with defaults
kn_sex_model = KNeighborsClassifier()
kn_sex_model.fit(X_sex_train_scaled, y_sex_train)

y_sex_pred_kn = kn_sex_model.predict(X_sex_validate_scaled)

print(classification_report(y_sex_validate, y_sex_pred_kn))

value_results(model=kn_sex_model, name="KNeighborsClassifier+Clean_SEX", model_predict=y_sex_pred_kn, y_test=y_sex_validate)