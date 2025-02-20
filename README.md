# Overstated Predictive Performance of a Pain Sensitivity Biomarker

| Authors      | Ole Goltermann          | Tamas Spisak       | Christian Büchel         |
|:-------------|:---------------------------|:----------------------|:---------------------|
| **Contact**  | o.goltermann[@]uke.de  | tamas.spisak[@]uk-essen.de | buechel[@]uke.de  |

![Last update](https://img.shields.io/badge/last_update-February_18,_2025-green)
#

This repository contains code for the re-analysis of the paper: ["Predicting Individual Pain Sensitivity Using a Novel Cortical Biomarker Signature"](https://jamanetwork.com/journals/jamaneurology/fullarticle/2829261). The analysis includes the preprocessing and modeling of original data to evaluate predictive models for individual pain sensitivity class (low vs high) based on peak-alpha frequency (PAF) and corticomotor excitability (CME). Based on our results, as well as inconsistencies in the code, we wrote the following **Letter to the Editor**. In addition to this letter, this repository contains a list of deviations from the pre-registered protocol, inconsistencies in the original code, as well as further comments on the validity of the reported findings. Instructions on how to use our code are at the end of this README. 

---
*Chowdhury, Bi et al. (2025)1 evaluated a biomarker for pain sensitivity, combining peak alpha frequency and corticomotor excitability. The authors report “outstanding” performance (AUCvalidation set = 1.0, AUCtest set = 0.88) and conclude that this biomarker is robust, reproducible, and has substantial clinical translation potential. While we appreciate the well-designed study and accompanying open data and analysis code, we unfortunately identified two critical methodological issues that undermine these conclusions*

*First, the reported AUC of 1.0 in the "validation set" is misleading. It was derived from a non-independent subset (n=16) taken directly from the training set (n=80), meaning the model was tested on data it had already seen. The AUC for such a subset does not yield any additional value and, in principle, it should match the training set’s AUC (=0.73). However, due to the small sample size, it is highly sensitive to the specific individuals sampled. Upon re-running the analysis code, it turned out that an extremely unrepresentative sample was chosen (by fixing the seed used for random sampling at 23), with 99.6% of possible samples yielding worse performance (see Figure 1A). This highlights that the reported AUC of 1.0 is not a representative measure of model performance but rather the consequence of a particular random seed.*

*Second, the AUC of 0.88 for the test set was again based on a single, randomly chosen train-test split. Although we acknowledge that the authors pre-registered the use of a single random split, the small test sample size (n=38), again, introduces a large risk of sampling bias, potentially inflating performance estimates and limiting generalizability.2,3 To evaluate the representativeness of their split, we reanalyzed the data using repeated train-test splits while keeping all other analysis steps identical (for a detailed analysis notebook and further technical comments see our github repository4). This revealed  a substantially lower average predictive performance in the test set: AUC = 0.74, accuracy = 0.68, with 99% of train-test splits yielding an AUC lower than the reported AUC of 0.88 (Figure 1B), making it again an outlier rather than a robust performance estimate. This implies that nearly one in three individuals would be classified incorrectly by this biomarker, severely limiting its clinical utility and raising serious doubts about its potential for translation into clinical practice.*

---

## Two main concerns about the results reported in the paper

As outlined in the letter, we found two critical methodological issues:

1. The reported AUC of 1 for the validation set is not based on a true validation set and, addtionally, results from using a fixed random seed (=23), which led to an extremely unrepresentative AUC estimate.
  
2. The reported AUC of 0.88 for the test set is based on a single, random train-test split. Given the small overall sample size for this type of analysis, the result is likely susceptible to biases introduced by specific sample characteristics. We found that only 10 out of 1000 iterations produced an AUC as high as 0.88. The probability of observing an AUC of 0.59 or lower is just as high as obtaining the AUC reported in the paper. 


   <img src="figures/Metrics_testset_histogram_with_cutoff.svg" alt="histograms" width="1200">


**Figure 1**. **(A)** Out of 1,000 sub-samples of 16 individuals taken from the training set (questionably  labeled as ‘validation’ set), only 4 result in an AUC of 1.0 (p=0.004). The mean AUC for this sub-sample is 0.73, matching the cross-validated AUC of the training set, as depicted in the histogram. (B) Out of 1,000 repeated analyses, in which all analysis steps - including the latent growth modelling - were identical to those in the original study, only 10 produced an AUC as high as the one reported in the paper (p=0.01). The joint probability of observing these two metrics is 0.004% (1 out of 25,000).

## Deviations from their [protocol](https://journals.lww.com/painrpts/fulltext/2020/08000/a_novel_cortical_biomarker_signature_for.6.aspx)

Listed below are only deviations that have an impact on the reported metrics, beyond the two critical issues mentioned above.

<table>
  <thead style="background-color: #808080; color: white;">
    <tr>
      <th>Protocol</th>
      <th>What they actually did</th>
      <th>Influence on results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Corticomotor excitability on Day 5 (measured as map volume, a continuous variable) as a predictor for the model</td>
      <td>The difference in map volume between Day 0 and Day 5 was calculated and then binarized.</td>
      <td>The original predictor (map volume as a continuous variable) and the non-binarized difference performed worse.</td>
    </tr>
    <tr>
      <td>"The sample of 150 subjects will first be randomly divided into an outer-training set (n = 100) and an outer-testing set (n = 50). The ratios of high- vs low-pain sensitive individuals will be matched between the 2 cohorts" (...) </td>
      <td> Test set is imbalanced. 24 are coded as high, 14 as low pain sensitivity. </td>
      <td> As demonstrated above, the AUC of the test set is unrepresentative and a clear outlier. </td>
    </tr>
  </tbody>
</table>

Adhering to the original pre-registered analysis, we got an AUC = `0.72`, accuracy = `0.68` for the training set and AUC = `0.71`, accuracy = `0.68` for the test set.

## Inconsistencies in the code   

We only report inconsistencies here that have an impact on model performance indices. 

Original Script is [here](https://github.com/DrNahianC/PREDICT_Scripts/blob/main/PREDICT_Scripts-main/Machine%20Learning%20Scripts/ML_classification_PAF_CME.py)

1. **Parameter tuning for Models**: 

     ClassifierTuner is imported, but not used for hyperparamter tuning at all
   
     <img src="figures/code_screenshots/ML_class_1.png" alt="ML_class_1" width="1200">

2. **Imputing data for training set**:

    Data is imputed in the trainings set for the whole set. That leads to information leakage between the k-folds. 
    
    <img src="figures/code_screenshots/ML_class_11.png" alt="ML_class_5" width="1200">

    
3. **Parameter space for Classifiers**
    
    Logistic regression: Maximizing maxiter is not a real hyperparameter for logistic regression. 

    <img src="figures/code_screenshots/ML_class_2.png" alt="ML_class_2" width="1200">

    Gradient Boosting: Learning rate 0.01 is used twice, Max depth of `None` doesn't make sense here

    <img src="figures/code_screenshots/ML_class_3.png" alt="ML_class_3" width="1200">

    Support Vector Classifier: Optimizing `gamma` doesn't make sense with `kernel='linear'`

    <img src="figures/code_screenshots/ML_class_4.png" alt="ML_class_4" width="1200">

    Mulit-layer Perceptron: alpha=0 is incorrect, it should always be positive; learning rate options could be added here

    <img src="figures/code_screenshots/ML_class_5.png" alt="ML_class_5" width="1200">
      
4. **AUC/Accuracy for so-called validation set**

    The authors report an AUC and accuracy of 1 for the 'validation' set. However, as shown above, its not an actual validation set, but rather a small sub-sample of 16 individuals from the training set, using a particular seed (**=23**) to sample these individuals. Here the code, they use to generate this result.

   (1) Here predictors of the training set are assigned to `X` and outcome to `y`

    <img src="figures/code_screenshots/ML_class_6.png" alt="ML_class_6" width="1200">

   (2) Now (after model training and locking of model), 16 individuals are drawn from the training data (X,y) using a **fixed random seed** of **23** (as shown this leads to an extreme unrepresentative sample that produces an outlier).

    <img src="figures/code_screenshots/ML_class_7.png" alt="ML_class_7" width="1200">

   (3) AUC is calculated for this sub-sample an reported as validation set AUC. 

    <img src="figures/code_screenshots/ML_class_8.png" alt="ML_class_8" width="1200">

   (4) The correct AUC to report would be either the average AUC across the 5 folds (**0.65**) or the performance of the locked model in the training set (**0.73**). This is done already here:

    <img src="figures/code_screenshots/ML_class_9.png" alt="ML_class_9" width="1200">

    We would like to note that it is somewhat unusal that the authors did not report the AUCs already calculated for their 5 folds (internal validation sets), which would have accurately reflect the performance of the models in their particular analysis. Instead, they opted to draw 16 individuals again, using a fixed random seed (=23).

## Further comments 

- The paper reports a wide range of sensitivity analyses. However, it is important to note that these do not affect the issues we raised above. For all their additional analyses, the calculated AUC and accuracy will drop significantly as well.
- Accroding to their protocol, the data was randomly split into training and test sets. We briefly tried to replicate the employed split, using the train_test_split() function from scikit-learn, which they also employed elsewhere in their scripts. However, we could not replicate the exact split with random seeds between 1 and 10,000,000. It would be interesting to know how the split was generated.

## Our analysis pipeline  

To reduce the risk of sample characteristic biases on the models performance estimate, we integrated a repeated split into training and test data. Additonally, we do not report metrics for a so called "validation set" but rather only report averaged metrics for training and test data with identical model configuration and sample sizes for the two sets as in the original paper. Our pipeline does the following:

1. **Split data in training and test set**: Sample size for training and test set is the same as in the original study
2. **Run the latent growth model** (based on the Rscript of the original paper) that classifies subjects into low and high pain sensitivity and uses this model for the test set classification as well (this is the "ground truth" for the later classification task)
3. **Model Training and Evaluation**: Trains several machine learning models (e.g., Logistic Regression, Random Forest, Gradient Boosting, SVC, MLPClassifier) 
4. **Result Aggregation**: Aggregates results across multiple random seeds and saves them to CSV files.
5. **Visualization**: Generates bar plots for accuracy and AUC metrics for each model.

## Requirements

To run this code, you need the following Python libraries:
- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib

Ensure you have Python 3.8 or higher installed.

### Installation
You can install the required libraries using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```txt
numpy
pandas
scikit-learn
seaborn
matplotlib
openpyxl
```

## How to Run the Code

1. Clone this repository:

   ```bash
   git clone https://github.com/your_username/PAF_reanalysis.git
   cd PAF_reanalysis
   ```

2. Run the Jupyter notebook

3. Execute the cells in order, ensuring the data files are in the specified `basepath`. The path is determined automatically. 

4. Results will be saved in CSV format:
   - `results_all_runs.csv`: Contains detailed performance metrics for each random seed and model.
   - `summary_results.csv`: Contains aggregated accuracy and AUC metrics for training and test datasets.

5. Visualization outputs are saved as SVG files:
   - `Accuracy_by_Model.svg`
   - `AUC_by_Model.svg`

## Results

AUC and accuracy for the five different models are displayed here (based on 1000 repetitions). 

<img src="figures/AUC_by_model.svg" alt="AUC" width="600">
<img src="figures/Accuracy_by_model.svg" alt="AUC" width="600">

## Contact
For questions or issues, please contact [ole.goltermann@maxplanckschools.de].

