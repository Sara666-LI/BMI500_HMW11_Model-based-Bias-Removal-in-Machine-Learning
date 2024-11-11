HW3: Model-based Bias Removal in Machine Learning using Synthetic Blood Pressure Data

Name: Liping Li       Email: liping.li@emory.edu



Code link on GitHub https://github.com/Sara666-LI/BMI500_HMW11_Model-based-Bias-Removal-in-Machine-Learning/



i. Choose both Polynomial Regression Model and Sigmoid-Gaussian Model for SBP and DBP here in my work.

ii. Implement the models in Python. Codes are on GitHub. 



Fitted Parameters for each model:

SBP Polynomial Model:

c1 = -0.002022 mmHg/year²

c2 = 0.498886 mmHg/year

c3 = 108.854700 mmHg



DBP Polynomial Model:

d1 = -0.005561 mmHg/year²

d2 = 0.583338 mmHg/year

d3 = 63.111150 mmHg



SBP Sigmoid Model:

Smax = 145.26 mmHg

k = 0.0198 year⁻¹

a0 = -53.81 years



DBP Gaussian Model:

Dmax = 78.50 mmHg

a_peak = 52.28 years

σ = 81.14 years



iii. The model curves are shown below:



iv. Parameter Interpretation:

Polynomial Model:

c₁ (SBP curvature): Physical dimension is mmHg/year², it represents the acceleration of SBP increase with age. Positive value indicates upward curvature

d₁ (DBP curvature): Physical dimension is mmHg/year². It captures the inverted-U shape of DBP. Negative value shows downward curvature



Sigmoidal-Gaussian Model:

Smax (~140 mmHg): Maximum SBP plateau in elderly. It physiologically represents maximum arterial stiffening, and is aligned with clinical hypertension thresholds

a₀ (~50 years): Age at half-maximum SBP rise. It indicates mid-life BP transition point, and correlates with cardiovascular risk increase

Dmax (~80 mmHg): Peak DBP in middle age, it represents optimal vascular resistance and physiologically meaningful threshold

apeak (~45 years): Age of maximum DBP. It corresponds to vascular elasticity peak, it’s an important clinical monitoring point

σ (~20 years): Spread of DBP change which indicates transition speed and reflects population variability



v. Discussion and Analysis:

i. For SBP, Sigmoid model is superior because it better captures physiological plateauing and more realistic at age extremes with better R² and MSE values.

For DBP, Gaussian model is better because it better fits observed data pattern, it naturally represents mid-life peak and physiologically plausible decline

ii. For SBP changes: Gradual increase reflects arterial stiffening, plateau represents maximal vascular changes, and rate parameter indicates progression speed.

For DBP changes: Mid-life peak shows vascular resistance, decline reflects arterial compliance loss and width indicates transition period



iii. Limitations:

The model's limitations in capturing demographic nuances include an inability to account for gender, ethnic, and socioeconomic differences. It also excludes key health-related factors like medications, comorbidities, and lifestyle influences. Temporal aspects are limited due to its cross-sectional nature, neglecting longitudinal data, secular trends, and seasonal variations. Statistically, the model assumes population homogeneity, underrepresents certain age ranges, and struggles with uncertainty at extreme ages. These gaps restrict its ability to generalize across diverse populations.





i.) Synthetic Blood Pressure Data Generation

Here, I created synthetic BP data using bivariate normal distributions for males and females and different male-to-female ratios while maintaining total sample size of 100000, which can be implemented by the function np.random.multivariate_normal in NumPy library. Then a binary label indicating male (1) or female (0) was assigned to each data.

ii.) Binary Classification

Divide the dataset into training (80%) and testing (20%) sets and then I chose logistic regression model as the binary classifier to estimate sex based on SBP and DBP values,

After trained the logistic regression classifier on datasets with varying male-to-female ratios (M and F). I got the model performance metrics (F1 score and accuracy) shown below. 





Alos, I plot the ROC curve for each model below.

 

Analysis:

i.  Classifier performance varies significantly with changes in male-to-female ratios. Balanced datasets (50% male, 50% female) yield baseline performance with an AUC of ~0.75. Male-dominated datasets (70–80% male) show increased male sensitivity but reduced female specificity, slightly lowering the AUC to ~0.72. Female-dominated datasets (20–30% male) improve female accuracy but reduce male performance, with an AUC close to the balanced case (~0.74). F1 scores are most stable with ratios between 40–60%, while extreme ratios (e.g., 20% or 80%) cause up to a 15% decrease and increased gender disparities. Class-specific performance peaks with balanced representation, with male classification suffering in female-dominated datasets and female accuracy declining sharply in male-dominated datasets.



ii.  Biases in the model arise due to imbalanced datasets, significantly impacting fairness and reliability. Prevalence-induced biases include majority class bias, where the model over-predicts the dominant group, causing higher false positives and reduced true positives for minorities, and underrepresentation bias, which weakens sensitivity and decision boundaries for the minority class. Clinically, these biases can lead to over-diagnosis in majority groups, missed diagnoses in minorities, and inappropriate treatment recommendations, ignoring gender-specific risks. Statistically, imbalances shift decision boundaries toward the majority class and distort feature importance, overemphasizing majority patterns while undervaluing minority characteristics. To mitigate biases, data-level solutions include stratified sampling, synthetic data generation (e.g., SMOTE), and ensuring minimum representation thresholds. Algorithm-level strategies involve applying class weights inversely proportional to class frequencies, using demographic-specific thresholds, and leveraging balanced ensemble methods. Evaluation-level approaches emphasize monitoring per-class performance metrics, using stratified cross-validation, and conducting regular bias audits to ensure fairness and reliability across demographic groups.





iii.)  Discussion:

i. Impact of Dataset Imbalance: 

Balanced datasets are essential in healthcare to ensure patient safety, care quality, and health equity. They improve diagnostic accuracy by representing diverse demographic groups, reducing missed diagnoses, and enabling accurate risk stratification. In treatment, they enhance outcome predictions, medication dosing, and personalized care. For clinical decision support, balanced data mitigate algorithmic bias, improving automated tools and triage accuracy. Health equity benefits include identifying disparities, enabling equitable screening, and ensuring fair resource allocation. They also support better understanding of disease patterns, more effective public health interventions, and improved preventive care strategies across populations.



ii. Real-world challenges with unbalanced datasets arise from both data collection and statistical issues. Data collection difficulties include demographic disparities, clinical practice variations, and resource constraints, leading to underrepresentation of certain populations, inconsistent data, and limited resources. Statistically, model performance is hindered by reduced accuracy for minority groups, overfitting to the majority, and difficulty in validation. Additionally, feature selection may bias models towards majority group characteristics, and threshold selection becomes challenging due to differing optimal thresholds across groups, affecting sensitivity and specificity.



iii. Practical strategies to address challenges in unbalanced datasets focus on both data collection and model development. Data collection approaches include active sampling strategies like targeted recruitment and oversampling, synthetic data generation using techniques like SMOTE, and ensuring data quality through standardized protocols and audits. For model development, strategies involve selecting algorithms robust to class imbalance, using ensemble and cost-sensitive methods, applying stratified cross-validation and class weight adjustments in training, and evaluating with per-class performance and fairness metrics to monitor and mitigate bias.





iv.) Bias Mitigation in Training:

Here I used Custom Bias-Aware Loss Function to modify the training strategy. Custom Bias-Aware Loss Function combined standard binary cross-entropy for classification accuracy and bias penalty term measuring male/female performance difference, it’s controlled by alpha parameter (higher alpha means stronger bias mitigation).



There are three key components of the strategy:

1. Class-Specific Loss Tracking: Separately tracks performance for males and females, and enables targeted optimization for each group.



male_mask = tf.cast(y_true == 1, tf.float32)

female_mask = tf.cast(y_true == 0, tf.float32)



2.Bias penalty term penalizes differences in performance between genders, it encourages model to maintain similar accuracy across groups.



Copybias_penalty = tf.abs(male_avg - female_avg)



3. Balanced class weights compensates for imbalanced class distribution, which ives higher importance to minority class samples



class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)



Reasoning Behind the Approach:

The approach leverages a custom loss function for direct fairness optimization, offering control over bias-accuracy trade-offs while maintaining performance. Class-specific tracking provides detailed bias monitoring, enabling targeted improvements and transparency. Balanced weights address data imbalance, preventing majority class dominance and enhancing minority class representation.









The results show that increasing the bias penalty (alpha) reduces the accuracy gap between males and females with the dataset of 100000, though potentially at the cost of overall accuracy. This approach allows practitioners to tune the bias-accuracy trade-off according to their specific needs.

