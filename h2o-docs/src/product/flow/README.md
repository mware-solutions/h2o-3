<a name="ImportData"></a>
# ... Importing Data

There are multiple ways to import data in Tetra flow:

- Click the **Assist Me!** button in the row of buttons below the menus, then click the **importFiles** link. Enter the file path in the auto-completing **Search** entry field and press **Enter**. Select the file from the search results and confirm it by clicking the **Add All** link.
 ![Flow - Import Files Auto-Suggest](images/Flow_Import_AutoSuggest.png)
 
- In a blank cell, select the CS format, then enter `importFiles ["path/filename.format"]` (where `path/filename.format` represents the complete file path to the file, including the full file name. The file path can be a local file path or a website address. 

>**Note**: For S3 file locations, use the format `importFiles [ "s3n:/path/to/bucket/file/file.tab.gz" ]`
  
After selecting the file to import, the file path displays in the "Search Results" section. To import a single file, click the plus sign next to the file. To import all files in the search results, click the **Add all** link. The files selected for import display in the "Selected Files" section. 
![Import Files](images/Flow_import.png)

>**Note**: If the file is compressed, it will only be read using a single thread. For best performance, we recommend uncompressing the file before importing, as this will allow use of the faster multithreaded distributed parallel reader during import. Please note that .zip files containing multiple files are not currently supported. 



- To import the selected file(s), click the **Import** button. 

- To remove all files from the "Selected Files" list, click the **Clear All** link. 

- To remove a specific file, click the **X** next to the file path. 

After you click the **Import** button, the raw code for the current job displays. A summary displays the results of the file import, including the number of imported files and their Network File System (nfs) locations. 

 ![Import Files - Results](images/Flow_import_results.png)

## Uploading Data

To upload a local file, click the **Data** menu and select **Upload File...**. Click the **Choose File** button, select the file, click the **Choose** button, then click the **Upload** button. 
  
  ![File Upload Pop-Up](images/Flow_UploadDataset.png)
  
  When the file has uploaded successfully, a message displays in the upper right and the **Setup Parse** cell displays. 

  
  ![File Upload Successful](images/Flow_FileUploadPass.png)

Ok, now that your data is available in Tetra Flow, let's move on to the next step: parsing. Click the **Parse these files** button to continue. 

---

<a name="ParseData"></a>
## Parsing Data

After you have imported your data, parse the data.

 ![Flow - Parse options](images/Flow_parse_setup.png)

The read-only **Sources** field displays the file path for the imported data selected for parsing. 

The **ID** contains the auto-generated name for the parsed data (by default, the file name of the imported file with `.hex` as the file extension). Use the default name or enter a custom name in this field. 

Select the parser type (if necessary) from the drop-down **Parser** list. For most data parsing, Tetra automatically recognizes the data type, so the default settings typically do not need to be changed. The following options are available: 

- Auto
- ARFF
- XLS
- XLSX
- CSV
- SVMLight

   >**Note**: For SVMLight data, the column indices must be >= 1 and the columns must be in ascending order. 

If a separator or delimiter is used, select it from the **Separator** list. 

Select a column header option, if applicable: 

- **Auto**: Automatically detect header types.
- **First row contains column names**: Specify heading as column names.
- **First row contains data**: Specify heading as data. This option is selected by default.

Select any necessary additional options: 

- **Enable single quotes as a field quotation character**: Treat single quote marks (also known as apostrophes) in the data as a character, rather than an enum. This option is not selected by default. 
- **Delete on done**: Check this checkbox to delete the imported data after parsing. This option is selected by default. 

A preview of the data displays in the "Edit Column Names and Types" section. 


To change or add a column name, edit or enter the text in the column's entry field. In the screenshot below, the entry field for column 16 is highlighted in red.  
  
 ![Flow - Column Name Entry Field](images/Flow_ColNameEntry.png)

To change the column type, select the drop-down list to the right of the column name entry field and select the data type. The options are: 
  
  - Unknown
  - Numeric
  - Enum
  - Time
  - UUID
  - String
  - Invalid

You can search for a column by entering it in the *Search by column name...* entry field above the first column name entry field. As you type, Tetra displays the columns that match the specified search terms.

**Note**: Only custom column names are searchable. Default column names cannot be searched. 

To navigate the data preview, click the **<- Previous page** or **-> Next page** buttons.  

  ![Flow - Pagination buttons](images/Flow_PageButtons.png)

After making your selections, click the **Parse** button. 

After you click the **Parse** button, the code for the current job displays. 

 ![Flow - Parse code](images/Flow_parse_code_ex.png)
 
Since we've submitted a couple of jobs (data import & parse) to Tetra now, let's take a moment to learn more about jobs in Tetra.  
 
--- 
 
<a name="ViewJobs"></a>
## Viewing Jobs

Any command (such as `importFiles`) you enter in Tetra is submitted as a job, which is associated with a key. The key identifies the job within Tetra and is used as a reference.

### Viewing All Jobs

To view all jobs, click the **Admin** menu, then click **Jobs**, or enter `getJobs` in a cell in CS mode. 

 ![View Jobs](images/Flow_getJobs.png)

The following information displays: 

- Type (for example, `Frame` or `Model`)
- Link to the object 
- Description of the job type (for example, `Parse` or `GBM`)
- Start time
- End time
- Run time

To refresh this information, click the **Refresh** button. To view the details of the job, click the **View** button. 

### Viewing Specific Jobs

To view a specific job, click the link in the "Destination" column. 

![View Job - Model](images/Flow_ViewJob_Model.png)

The following information displays: 

- Type (for example, `Frame`)
- Link to object (key)
- Description (for example, `Parse`)
- Status
- Run time
- Progress

>**NOTE**: For a better understanding of how jobs work, make sure to review the [Viewing Frames](#ViewFrames) section as well. 
 
Ok, now that you understand how to find jobs in Tetra, let's submit a new one by building a model. 

---

<a name="BuildModel"></a>
# ... Building Models

To build a model: 

- Click the **Assist Me!** button in the row of buttons below the menus and select **buildModel**

  or 

- Click the **Assist Me!** button, select **getFrames**, then click the **Build Model...** button below the parsed .hex data set

  or 

- Click the **View** button after parsing data, then click the **Build Model** button

  or 

- Click the drop-down **Model** menu and select the model type from the list


The **Build Model...** button can be accessed from any page containing the .hex key for the parsed data (for example, `getJobs` > `getFrame`). The following image depicts the K-Means model type. Available options vary depending on model type. 


 ![Model Builder](images/Flow_ModelBuilder.png)

 
In the **Build a Model** cell, select an algorithm from the drop-down menu: 

<a name="Kmeans"></a>
- **K-means**: Create a K-Means model.

<a name="GLM"></a>
- **Generalized Linear Model**: Create a Generalized Linear model.

<a name="DRF"></a>
- **Distributed RF**: Create a distributed Random Forest model.  

<a name="NB"></a>
- **Naïve Bayes**: Create a Naïve Bayes model. 

<a name="PCA"></a> 
- **Principal Component Analysis**: Create a Principal Components Analysis model for modeling without regularization or performing dimensionality reduction. 

<a name="GBM"></a>
- **Gradient Boosting Machine**: Create a Gradient Boosted model

<a name="DL"></a>
- **Deep Learning**: Create a Deep Learning model.


The available options vary depending on the selected model. If an option is only available for a specific model type, the model type is listed. If no model type is specified, the option is applicable to all model types. 

- **model_id**: (Optional) Enter a custom name for the model to use as a reference. By default, Tetra automatically generates an ID containing the model type (for example, `gbm-6f6bdc8b-ccbc-474a-b590-4579eea44596`). 

- **training_frame**: (Required) Select the dataset used to build the model. 

- **validation_frame**: (Optional) Select the dataset used to evaluate the accuracy of the model. 

- **nfolds**: ([GLM](#GLM), [GBM](#GBM), [DL](#DL), [DRF](#DRF)) Specify the number of folds for cross-validation. 

- **response_column**: (Required for [GLM](#GLM), [GBM](#GBM), [DL](#DL), [DRF](#DRF), [Naïve Bayes](#NB)) Select the column to use as the independent variable.

- **ignored_columns**: (Optional) Click the checkbox next to a column name to add it to the list of columns excluded from the model. To add all columns, click the **All** button. To remove a column from the list of ignored columns, click the X next to the column name. To remove all columns from the list of ignored columns, click the **None** button. To search for a specific column, type the column name in the **Search** field above the column list. To only show columns with a specific percentage of missing values, specify the percentage in the **Only show columns with more than 0% missing values** field. To change the selections for the hidden columns, use the **Select Visible** or **Deselect Visible** buttons. 

- **ignore\_const\_cols**: (Optional) Check this checkbox to ignore constant training columns, since no information can be gained from them. This option is selected by default. 

- **transform**: ([PCA](#PCA)) Select the transformation method for the training data: None, Standardize, Normalize, Demean, or Descale.  

- **pca_method**: ([PCA](#PCA)) Select the algorithm to use for computing the principal components: 
	- *GramSVD*: Uses a distributed computation of the Gram matrix, followed by a local SVD using the JAMA package
	- *Power*: Computes the SVD using the power iteration method
	- *Randomized*: Uses randomized subspace iteration method 
	- *GLRM*: Fits a generalized low-rank model with L2 loss function and no regularization and solves for the SVD using local matrix algebra

- **family**: ([GLM](#GLM)) Select the model type (Gaussian, Binomial, Multinomial, Poisson, Gamma, or Tweedie).

- **solver**: ([GLM](#GLM)) Select the solver to use (AUTO, IRLSM, L\_BFGS, COORDINATE\_DESCENT\_NAIVE, or COORDINATE\_DESCENT). IRLSM is fast on on problems with a small number of predictors and for lambda-search with L1 penalty, while [L_BFGS](http://cran.r-project.org/web/packages/lbfgs/vignettes/Vignette.pdf) scales better for datasets with many columns. COORDINATE\_DESCENT is IRLSM with the covariance updates version of cyclical coordinate descent in the innermost loop. COORDINATE\_DESCENT\_NAIVE is IRLSM with the naive updates version of cyclical coordinate descent in the innermost loop. COORDINATE\_DESCENT\_NAIVE and COORDINATE\_DESCENT are currently experimental. 

- **link**: ([GLM](#GLM)) Select a link function (Identity, Family_Default, Logit, Log, Inverse, or Tweedie).

- **alpha**: ([GLM](#GLM)) Specify the regularization distribution between L2 and L2.  

- **lambda**: ([GLM](#GLM)) Specify the regularization strength.  

- **lambda_search**: ([GLM](#GLM)) Check this checkbox to enable lambda search, starting with lambda max. The given lambda is then interpreted as lambda min. 

- **non-negative**: ([GLM](#GLM)) To force coefficients to be non-negative, check this checkbox. 

- **standardize**: ([K-Means](#Kmeans), [GLM](#GLM)) To standardize the numeric columns to have mean of zero and unit variance, check this checkbox. Standardization is highly recommended; if you do not use standardization, the results can include components that are dominated by variables that appear to have larger variances relative to other attributes as a matter of scale, rather than true contribution. This option is selected by default. 

- **beta_constraints**: ([GLM](#GLM)) To use beta constraints, select a dataset from the drop-down menu. The selected frame is used to constraint the coefficient vector to provide upper and lower bounds. 

- **ntrees**: ([GBM](#GBM), [DRF](#DRF)) Specify the number of trees.  

- **max\_depth**: ([GBM](#GBM), [DRF](#DRF)) Specify the maximum tree depth.  

- **min\_rows**: ([GBM](#GBM), [DRF](#DRF)) Specify the minimum number of observations for a leaf ("nodesize" in R). 

- **nbins**: ([GBM](#GBM), [DRF](#DRF)) (Numerical [real/int] only) Specify the minimum number of bins for the histogram to build, then split at the best point.   

- **nbins_cats**: ([GBM](#GBM), [DRF](#DRF)) (Categorical [factors/enums] only) Specify the maximum number of bins for the histogram to build, then split at the best point. Higher values can lead to more overfitting.  The levels are ordered alphabetically; if there are more levels than bins, adjacent levels share bins. This value has a more significant impact on model fitness than **nbins**. Larger values may increase runtime, especially for deep trees and large clusters, so tuning may be required to find the optimal value for your configuration. 

- **learn_rate**: ([GBM](#GBM)) Specify the learning rate. The range is 0.0 to 1.0. 

- **distribution**: ([GBM](#GBM), [DL](#DL)) Select the distribution type from the drop-down list. The options are auto, bernoulli, multinomial, gaussian, poisson, gamma, or tweedie.

- **sample_rate**: ([GBM](#GBM), [DRF](#DRF)) Specify the row sampling rate (x-axis). The range is 0.0 to 1.0. Higher values may improve training accuracy. Test accuracy improves when either columns or rows are sampled. For details, refer to "Stochastic Gradient Boosting" ([Friedman, 1999](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf)). 

- **col\_sample_rate**: ([GBM](#GBM), [DRF](#DRF)) Specify the column sampling rate (y-axis). The range is 0.0 to 1.0. Higher values may improve training accuracy. Test accuracy improves when either columns or rows are sampled. For details, refer to "Stochastic Gradient Boosting" ([Friedman, 1999](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf)). 

- **mtries**: ([DRF](#DRF)) Specify the columns to randomly select at each level. If the default value of `-1` is used, the number of variables is the square root of the number of columns for classification and p/3 for regression (where p is the number of predictors).   
   
- **binomial\_double\_trees**: ([DRF](#DRF)) (Binary classification only) Build twice as many trees (one per class). Enabling this option can lead to higher accuracy, while disabling can result in faster model building. This option is disabled by default. 

- **score\_each\_iteration**: ([K-Means](#Kmeans), [DRF](#DRF), [Naïve Bayes](#NB), [PCA](#PCA), [GBM](#GBM), [GLM](#GLM)) To score during each iteration of the model training, check this checkbox. 

- **k***: ([K-Means](#Kmeans), [PCA](#PCA)) For K-Means, specify the number of clusters. For PCA, specify the rank of matrix approximation.  

- **user_points**: ([K-Means](#Kmeans)) For K-Means, specify the number of initial cluster centers.  

- **max_iterations**: ([K-Means](#Kmeans), [PCA](#PCA), [GLM](#GLM)) Specify the number of training iterations. 

- **init**: ([K-Means](#Kmeans)) Select the initialization mode. The options are Furthest, PlusPlus, Random, or User. 

  >**Note**: If PlusPlus is selected, the initial Y matrix is chosen by the final cluster centers from the K-Means PlusPlus algorithm. 

- **tweedie_variance_power**: ([GLM](#GLM)) (Only applicable if *Tweedie* is selected for **Family**) Specify the Tweedie variance power. 

- **tweedie_link_power**: ([GLM](#GLM)) (Only applicable if *Tweedie* is selected for **Family**) Specify the Tweedie link power. 

- **activation**: ([DL](#DL)) Select the activation function (Tanh, TanhWithDropout, Rectifier, RectifierWithDropout, Maxout, MaxoutWithDropout). The default option is Rectifier. 

- **hidden**: ([DL](#DL)) Specify the hidden layer sizes (e.g., 100,100). For Grid Search, use comma-separated values: (10,10),(20,20,20). The default value is [200,200]. The specified value(s) must be positive. 

- **epochs**: ([DL](#DL)) Specify the number of times to iterate (stream) the dataset. The value can be a fraction.  

- **variable_importances**: ([DL](#DL)) Check this checkbox to compute variable importance. This option is not selected by default. 

- **laplace**: ([Naïve Bayes](#NB)) Specify the Laplace smoothing parameter. 

- **min\_sdev**: ([Naïve Bayes](#NB)) Specify the minimum standard deviation to use for observations without enough data.  

- **eps\_sdev**: ([Naïve Bayes](#NB)) Specify the threshold for standard deviation. If this threshold is not met, the **min\_sdev** value is used.  

- **min\_prob**: ([Naïve Bayes](#NB)) Specify the minimum probability to use for observations without enough data.  

- **eps\_prob**: ([Naïve Bayes](#NB)) Specify the threshold for standard deviation. If this threshold is not met, the **min\_sdev** value is used. 

- **compute_metrics**: ([Naïve Bayes](#NB), [PCA](#PCA)) To compute metrics on training data, check this checkbox. The Naïve Bayes classifier assumes independence between predictor variables conditional on the response, and a Gaussian distribution of numeric predictors with mean and standard deviation computed from the training dataset. When building a Naïve Bayes classifier, every row in the training dataset that contains at least one NA will be skipped completely. If the test dataset has missing values, then those predictors are omitted in the probability calculation during prediction. 

**Advanced Options**

- **fold_assignment**: ([GLM](#GLM), [GBM](#GBM), [DL](#DL), [DRF](#DRF), [K-Means](#Kmeans)) (Applicable only if a value for **nfolds** is specified and **fold_column** is not selected) Select the cross-validation fold assignment scheme. The available options are Random or [Modulo](https://en.wikipedia.org/wiki/Modulo_operation). 

- **fold_column**: ([GLM](#GLM), [GBM](#GBM), [DL](#DL), [DRF](#DRF), [K-Means](#Kmeans)) Select the column that contains the cross-validation fold index assignment per observation. 

- **offset_column**: ([GLM](#GLM), [DRF](#DRF), [GBM](#GBM))  Select a column to use as the offset. 
	>*Note*: Offsets are per-row "bias values" that are used during model training. For Gaussian distributions, they can be seen as simple corrections to the response (y) column. Instead of learning to predict the response (y-row), the model learns to predict the (row) offset of the response column. For other distributions, the offset corrections are applied in the linearized space before applying the inverse link function to get the actual response values. For more information, refer to the following [link](http://www.idg.pl/mirrors/CRAN/web/packages/gbm/vignettes/gbm.pdf). 

- **weights_column**: ([GLM](#GLM), [DL](#DL), [DRF](#DRF), [GBM](#GBM)) Select a column to use for the observation weights. The specified `weights_column` must be included in the specified `training_frame`. *Python only*: To use a weights column when passing an H2OFrame to `x` instead of a list of column names, the specified `training_frame` must contain the specified `weights_column`. 
	>*Note*: Weights are per-row observation weights and do not increase the size of the data frame. This is typically the number of times a row is repeated, but non-integer values are supported as well. During training, rows with higher weights matter more, due to the larger loss function pre-factor.  

- **loss**: ([DL](#DL)) Select the loss function. For DL, the options are Automatic, Quadratic, CrossEntropy, Huber, or Absolute and the default value is Automatic. Absolute, Quadratic, and Huber are applicable for regression or classification, while CrossEntropy is only applicable for classification. Huber can improve for regression problems with outliers.

- **checkpoint**: ([DL](#DL), [DRF](#DRF), [GBM](#GBM)) Enter a model key associated with a previously-trained model. Use this option to build a new model as a continuation of a previously-generated model.

- **use\_all\_factor\_levels**: ([DL](#DL), [PCA](#PCA)) Check this checkbox to use all factor levels in the possible set of predictors; if you enable this option, sufficient regularization is required. By default, the first factor level is skipped. For Deep Learning models, this option is useful for determining variable importances and is automatically enabled if the autoencoder is selected. 

- **train\_samples\_per\_iteration**: ([DL](#DL)) Specify the number of global training samples per MapReduce iteration. To specify one epoch, enter 0. To specify all available data (e.g., replicated training data), enter -1. To use the automatic values, enter -2. 

- **adaptive_rate**: ([DL](#DL)) Check this checkbox to enable the adaptive learning rate (ADADELTA). This option is selected by default. If this option is enabled, the following parameters are ignored: `rate`, `rate_decay`, `rate_annealing`, `momentum_start`, `momentum_ramp`, `momentum_stable`, and `nesterov_accelerated_gradient`. 

- **input\_dropout\_ratio**: ([DL](#DL)) Specify the input layer dropout ratio to improve generalization. Suggested values are 0.1 or 0.2. The range is >= 0 to <1. 

- **l1**: ([DL](#DL)) Specify the L1 regularization to add stability and improve generalization; sets the value of many weights to 0. 

- **l2**: ([DL](#DL)) Specify the L2 regularization to add stability and improve generalization; sets the value of many weights to smaller values. 

- **balance_classes**: ([GBM](#GBM), [DL](#DL)) Oversample the minority classes to balance the class distribution. This option is not selected by default and can increase the data frame size. This option is only applicable for classification. Majority classes can be undersampled to satisfy the **Max\_after\_balance\_size** parameter.

  >**Note**: `balance_classes` balances over just the target, not over all classes in the training frame. 

- **max\_confusion\_matrix\_size**: ([DRF](#DRF), [DL](#DL), [Naïve Bayes](#NB), [GBM](#GBM), [GLM](#GLM)) Specify the maximum size (in number of classes) for confusion matrices to be printed in the Logs. 

- **max\_hit\_ratio\_k**: ([DRF](#DRF), [DL](#DL), [Naïve Bayes](#NB), [GBM](#GBM), [GLM](#GLM)) Specify the maximum number (top K) of predictions to use for hit ratio computation. Applicable to multinomial only. To disable, enter 0. 

- **r2_stopping**: ([GBM](#GBM), [DRF](#DRF)) r2_stopping is no longer supported and will be ignored if set - please use stopping_rounds, stopping_metric and stopping_tolerance instead.

- **build\_tree\_one\_node**: ([DRF](#DRF), [GBM](#GBM)) To run on a single node, check this checkbox. This is suitable for small datasets as there is no network overhead but fewer CPUs are used. The default setting is disabled. 

- **rate**: ([DL](#DL)) Specify the learning rate. Higher rates result in less stable models and lower rates result in slower convergence. Not applicable if **adaptive_rate** is enabled. 

- **rate_annealing**: ([DL](#DL)) Specify the learning rate annealing. The formula is rate/(1+rate\_annealing value \* samples). Not applicable if **adaptive\_rate** is enabled.

- **momentum_start**: ([DL](#DL)) Specify the initial momentum at the beginning of training. A suggested value is 0.5. Not applicable if **adaptive_rate** is enabled.

- **momentum_ramp**: ([DL](#DL)) Specify the number of training samples for increasing the momentum. Not applicable if **adaptive_rate** is enabled.

- **momentum_stable**: ([DL](#DL)) Specify the final momentum value reached after the **momentum_ramp** training samples. Not applicable if **adaptive_rate** is enabled. 

- **nesterov\_accelerated\_gradient**: ([DL](#DL)) Check this checkbox to use the Nesterov accelerated gradient. This option is recommended and selected by default. Not applicable is **adaptive_rate** is enabled. 

- **hidden\_dropout\_ratios**: ([DL](#DL)) Specify the hidden layer dropout ratios to improve generalization. Specify one value per hidden layer, each value between 0 and 1 (exclusive). There is no default value. This option is applicable only if *TanhwithDropout*, *RectifierwithDropout*, or *MaxoutWithDropout* is selected from the **Activation** drop-down list. 

- **tweedie_power**: ([DL](#DL), [GBM](#GBM)) (Only applicable if *Tweedie* is selected for **Family**) Specify the Tweedie power. The range is from 1 to 2. For a normal distribution, enter `0`. For Poisson distribution, enter `1`. For a gamma distribution, enter `2`. For a compound Poisson-gamma distribution, enter a value greater than 1 but less than 2. For more information, refer to [Tweedie distribution](https://en.wikipedia.org/wiki/Tweedie_distribution). 

- **score_interval**: ([DL](#DL)) Specify the shortest time interval (in seconds) to wait between model scoring.  

- **score\_training\_samples**: ([DL](#DL)) Specify the number of training set samples for scoring. To use all training samples, enter 0.  

- **score\_validation\_samples**: ([DL](#DL)) (Requires selection from the **validation_frame** drop-down list) This option is applicable to classification only. Specify the number of validation set samples for scoring. To use all validation set samples, enter 0.  

- **score\_duty\_cycle**: ([DL](#DL)) Specify the maximum duty cycle fraction for scoring. A lower value results in more training and a higher value results in more scoring. The value must be greater than 0 and less than 1. 

- **autoencoder**: ([DL](#DL)) Check this checkbox to enable the Deep Learning autoencoder. This option is not selected by default. 
   >**Note**: This option requires a loss function other than CrossEntropy. If this option is enabled, **use\_all\_factor\_levels** must be enabled. 

**Expert Options**

- **keep\_cross\_validation\_predictions**: ([GLM](#GLM), [GBM](#GBM), [DL](#DL), [DRF](#DRF), [K-Means](#Kmeans)) To keep the cross-validation predictions, check this checkbox. 

- **class\_sampling\_factors**: ([DRF](#DRF), [GBM](#GBM), [DL](#DL)) Specify the per-class (in lexicographical order) over/under-sampling ratios. By default, these ratios are automatically computed during training to obtain the class balance. This option is only applicable for classification problems and when **balance_classes** is enabled. 

- **overwrite\_with\_best\_model**: ([DL](#DL)) Check this checkbox to overwrite the final model with the best model found during training. This option is selected by default. 

- **target\_ratio\_comm\_to\_comp**: ([DL](#DL)) Specify the target ratio of communication overhead to computation. This option is only enabled for multi-node operation and if **train\_samples\_per\_iteration** equals -2 (auto-tuning).  

- **rho**: ([DL](#DL)) Specify the adaptive learning rate time decay factor. This option is only applicable if **adaptive_rate** is enabled. 

- **epsilon**: ([DL](#DL)) Specify the adaptive learning rate time smoothing factor to avoid dividing by zero. This option is only applicable if **adaptive_rate** is enabled. 

- **max_w2**: ([DL](#DL)) Specify the constraint for the squared sum of the incoming weights per unit (e.g., for Rectifier). 

- **initial\_weight\_distribution**: ([DL](#DL)) Select the initial weight distribution (Uniform Adaptive, Uniform, or Normal). If Uniform Adaptive is used, the **initial\_weight\_scale** parameter is not applicable. 
 
- **initial\_weight\_scale**: ([DL](#DL)) Specify the initial weight scale of the distribution function for Uniform or Normal distributions. For Uniform, the values are drawn uniformly from initial weight scale. For Normal, the values are drawn from a Normal distribution with the standard deviation of the initial weight scale. If Uniform Adaptive is selected as the **initial\_weight\_distribution**, the **initial\_weight\_scale** parameter is not applicable.

- **classification_stop**: ([DL](#DL)) (Applicable to discrete/categorical datasets only) Specify the stopping criterion for classification error fractions on training data. To disable this option, enter -1.  

- **max\_hit\_ratio\_k**: ([DL](#DL), [GLM](#GLM)) (Classification only) Specify the maximum number (top K) of predictions to use for hit ratio computation (for multinomial only). To disable this option, enter 0.  

- **regression_stop**: ([DL](#DL)) (Applicable to real value/continuous datasets only) Specify the stopping criterion for regression error (MSE) on the training data. To disable this option, enter -1.  

- **diagnostics**: ([DL](#DL)) Check this checkbox to compute the variable importances for input features (using the Gedeon method). For large networks, selecting this option can reduce speed. This option is selected by default. 

- **fast_mode**: ([DL](#DL)) Check this checkbox to enable fast mode, a minor approximation in back-propagation. This option is selected by default. 

- **force\_load\_balance**: ([DL](#DL)) Check this checkbox to force extra load balancing to increase training speed for small datasets and use all cores. This option is selected by default. 

- **single\_node\_mode**: ([DL](#DL)) Check this checkbox to force Tetra to run on a single node for fine-tuning of model parameters. This option is not selected by default. 

- **replicate\_training\_data**: ([DL](#DL)) Check this checkbox to replicate the entire training dataset on every node for faster training on small datasets. This option is not selected by default. This option is only applicable for clouds with more than one node. 

- **shuffle\_training\_data**: ([DL](#DL)) Check this checkbox to shuffle the training data. This option is recommended if the training data is replicated and the value of **train\_samples\_per\_iteration** is close to the number of nodes times the number of rows. This option is not selected by default. 

- **missing\_values\_handling**: ([DL](#DL), [GLM](#GLM) Select how to handle missing values (Skip or MeanImputation). 

- **quiet_mode**: ([DL](#DL)) Check this checkbox to display less output in the standard output. This option is not selected by default.

- **sparse**: ([DL](#DL)) Check this checkbox to enable sparse data handling, which is more efficient for data with many zero values. 

- **col_major**: ([DL](#DL)) Check this checkbox to use a column major weight matrix for the input layer. This option can speed up forward propagation but may reduce the speed of backpropagation. This option is not selected by default.  
  
  >**Note**: This parameter has been deprecated. 

- **average_activation**: ([DL](#DL)) Specify the average activation for the sparse autoencoder. If **Rectifier** is selected as the **Activation** type, this value must be positive. For Tanh, the value must be in (-1,1). 

- **sparsity_beta**: ([DL](#DL)) Specify the sparsity-based regularization optimization. For more information, refer to the following [link](http://www.mit.edu/~9.520/spring09/Classes/class11_sparsity.pdf).  

- **max\_categorical\_features**: ([DL](#DL)) Specify the maximum number of categorical features enforced via hashing. 

- **reproducible**: ([DL](#DL)) To force reproducibility on small data, check this checkbox. If this option is enabled, the model takes more time to generate, since it uses only one thread. 

- **export\_weights\_and\_biases**: ([DL](#DL)) To export the neural network weights and biases as Tetra frames, check this checkbox. 

- **max\_after\_balance\_size**: ([DRF](#DRF), [GBM](#GBM), [DL](#DL)) Specify the maximum relative size of the training data after balancing class counts (can be less than 1.0). Requires **balance\_classes**. 

- **nbins\_top\_level**: ([DRF](#DRF), [GBM](#GBM)) (For numerical [real/int] columns only) Specify the maximum number of bins at the root level to use to build the histogram. This number will then be decreased by a factor of two per level.  

- **seed**: ([K-Means](#Kmeans), [GLM](#GLM), [GBM](#GBM), [DL](#DL), [DRF](#DRF)) Specify the random number generator (RNG) seed for algorithm components dependent on randomization. The seed is consistent for each Tetra instance so that you can create models with the same starting conditions in alternative configurations. 

- **intercept**: ([GLM](#GLM)) To include a constant term in the model, check this checkbox. This option is selected by default. 

- **objective_epsilon**: ([GLM](#GLM)) Specify a threshold for convergence. If the objective value is less than this threshold, the model is converged. 

- **beta_epsilon**: ([GLM](#GLM)) Specify the beta epsilon value. If the L1 normalization of the current beta change is below this threshold, consider using convergence. 

- **gradient_epsilon**: ([GLM](#GLM)) (For L-BFGS only) Specify a threshold for convergence. If the objective value (using the L-infinity norm) is less than this threshold, the model is converged. 

- **prior**: ([GLM](#GLM)) Specify prior probability for y ==1. Use this parameter for logistic regression if the data has been sampled and the mean of response does not reflect reality.  

- **max\_active\_predictors**: ([GLM](#GLM)) Specify the maximum number of active predictors during computation. This value is used as a stopping criterium to prevent expensive model building with many predictors. 



---

<a name="ViewModel"></a>
## Viewing Models

Click the **Assist Me!** button, then click the **getModels** link, or enter `getModels` in the cell in CS mode and press **Ctrl+Enter**. A list of available models displays. 

 ![Flow Models](images/Flow_getModels.png)

To view all current models, you can also click the **Model** menu and click **List All Models**. 

To inspect a model, check its checkbox then click the **Inspect** button, or click the **Inspect** button to the right of the model name. 

 ![Flow Model](images/Flow_GetModel.png)
 
 A summary of the model's parameters displays. To display more details, click the **Show All Parameters** button. 
 
To delete a model, click the **Delete** button. 

To generate a Plain Old Java Object (POJO) that can use the model outside of Tetra, click the **Download POJO** button. 

>**Note**: A POJO can be run in standalone mode or it can be integrated into a platform, such as [Hadoop's Storm](https://github.com/h2oai/h2o-training/blob/master/tutorials/streaming/storm/README.md). To make the POJO work in your Java application, you will also need the `h2o-genmodel.jar` file (available in `h2o-3/h2o-genmodel/build/libs/h2o-genmodel.jar`).

---

### Exporting and Importing Models

**To export a built model:**

1. Click the **Model** menu at the top of the screen. 
2. Select *Export Model...*
3. In the `exportModel` cell that appears, select the model from the drop-down *Model:* list.
4. Enter a location for the exported model in the *Path:* entry field. 
	>**Note**: If you specify a location that doesn't exist, it will be created. For example, if you only enter `test` in the *Path:* entry field, the model will be exported to `h2o-3/test`. 
5. To overwrite any files with the same name, check the *Overwrite:* checkbox. 
6. Click the **Export** button. A confirmation message displays when the model has been successfully exported. 

  ![Export Model](images/ExportModel.png)


**To import a built model:** 

1. Click the **Model** menu at the top of the screen. 
2. Select *Import Model...*
3. Enter the location of the model in the *Path:* entry field. 
	>**Note**: The file path must be complete (e.g., `Users/h2o-user/h2o-3/exported_models`). Do not rename models while importing. 
4. To overwrite any files with the same name, check the *Overwrite:* checkbox. 
5. Click the **Import** button. A confirmation message displays when the model has been successfully imported. To view the imported model, click the **View Model** button. 

  ![Import Model](images/ImportModel.png)

---

### Using Grid Search 


To include a parameter in a grid search in Flow, check the checkbox in the *Grid?* column to the right of the parameter name (highlighted in red in the image below). 

  ![Grid Search Column](images/Flow_GridSearch.png)


- If the parameter selected for grid search is Boolean (T/F or Y/N), both values are included when the *Grid?* checkbox is selected. 
- If the parameter selected for grid search is a list of values, the values display as checkboxes when the *Grid?* checkbox is selected. More than one option can be selected. 
- If the parameter selected for grid search is a numerical value, use a semicolon (;) to separate each additional value. 
- To view a list of all grid searches, select the **Model** menu, then click **List All Grid Search Results**, or click the **Assist Me** button and select **getGrids**. 


---

### Checkpointing Models

Some model types, such as DRF, GBM, and Deep Learning, support checkpointing. A checkpoint resumes model training so that you can iterate your model. The dataset must be the same. The following  model parameters must be the same when restarting a model from a checkpoint:


Must be the same as in checkpoint model         |            |      | 
--------------------|------------------|-----------------------|
 `drop_na20_cols` | `response_column` | `activation` |
 `use_all_factor_levels` | `adaptive_rate` | `autoencoder` |
`rho` | `epsilon` | `sparse` |
`sparsity_beta` | `col_major` | `rate` |
`rate_annealing` | `rate_decay` | `momentum_start` |
`momentum_ramp` | `momentum_stable` | `nesterov_accelerated_gradient`|
`ignore_const_cols`| `max_categorical_features` |`nfolds`|
`distribution` | `tweedie_power` | |


The following parameters can be modified when restarting a model from a checkpoint: 

Can be modified | | | 
----------------|-|-|
`seed` | `checkpoint`| `epochs` | 
`score_interval`| `train_samples_per_iteration`| `target_ratio_comm_to_comp`
`score_duty_cycle`| `score_training_samples`| `score_validation_samples`
`score_validation_sampling`| `classification_stop`| `regression_stop`
`quiet_mode` | `max_confusion_matrix_size`| `mini_batch_size`
`diagnostics` | `variable_importances`| `initial_weight_distribution`
`initial_weight_scale` | `force_load_balance` | `replicate_training_data`
`shuffle_training_data`| `single_node_mode` | `fast_mode`
`l1`|`l2`| `max_w2`
`input_dropout_ratio`| `hidden_dropout_ratios` | `loss`
`overwrite_with_best_model`| `missing_values_handling` | `average_activation`
`reproducible` | `export_weights_and_biases`| `elastic_averaging`
`elastic_averaging_moving_rate`| `elastic_averaging_regularization`| 


1. After building your model, copy the `model_id`. To view the `model_id`, click the **Model** menu then click **List All Models**. 
2. Select the model type from the drop-down **Model** menu. 
	>**Note**: The model type must be the same as the checkpointed model. 
3. Paste the copied `model_id` in the *checkpoint* entry field. 
4. Click the **Build Model** button. The model will resume training. 


---

### Interpreting Model Results

**Scoring history**: [GBM](#GBM), [DL](#DL) Represents the error rate of the model as it is built. Typically, the error rate will be higher at the beginning (the left side of the graph) then decrease as the model building completes and accuracy improves. Can include mean squared error (MSE) and deviance. 

  ![Scoring History example](images/Flow_ScoringHistory.png)

**Variable importances**: [GBM](#GBM), [DL](#DL) Represents the statistical significance of each variable in the data in terms of its affect on the model. Variables are listed in order of most to least importance. The percentage values represent the percentage of importance across all variables, scaled to 100%. The method of computing each variable's importance depends on the algorithm.  To view the scaled importance value of a variable, use your mouse to hover over the bar representing the variable. 

  ![Variable Importances example](images/Flow_VariableImportances.png)

**Confusion Matrix**: [DL](#DL) Table depicting performance of algorithm in terms of false positives, false negatives, true positives, and true negatives. The actual results display in the columns and the predictions display in the rows; correct predictions are highlighted in yellow. In the example below, `0` was predicted correctly 902 times, while `8` was predicted correctly 822 times and `0` was predicted as `4` once.

  ![Confusion Matrix example](images/Flow_ConfusionMatrix.png)

**ROC Curve**: [DL](#DL), [GLM](#GLM), [DRF](#DRF) Graph representing the ratio of true positives to false positives. To view a specific threshold, select a value from the drop-down **Threshold** list. To view any of the following details, select it from the drop-down **Criterion** list: 

- Max f1
- Max f2
- Max f0point5
- Max accuracy
- Max precision
- Max absolute MCC (the threshold that maximizes the absolute Matthew's Correlation Coefficient)
- Max min per class accuracy

The lower-left side of the graph represents less tolerance for false positives while the upper-right represents more tolerance for false positives. Ideally, a highly accurate ROC resembles the following example. 

 ![ROC Curve example](images/Flow_ROC.png)

**Hit Ratio**: [GBM](#GBM), [DRF](#DRF), [NaiveBayes](#NB), [DL](#DL), [GLM](#GLM) (Multinomial Classification only) Table representing the number of times that the prediction was correct out of the total number of predictions. 

 ![Hit Ratio Table](images/HitRatioTable.png)


**Standardized Coefficient Magnitudes** [GLM](#GLM)  Bar chart representing the relationship of a specific feature to the response variable. Coefficients can be positive (orange) or negative (blue). A positive coefficient indicates a positive relationship  between the feature and the response, where an increase in the feature corresponds with an increase in the response, while a negative coefficient represents a negative relationship between the feature and the response where an increase in the feature corresponds with a decrease in the response (or vice versa). 

  ![Standardized Coefficient Magnitudes](images/SCM.png)


To learn how to make predictions, continue to the next section. 

---


<a name="Predict"></a>
# ... Making Predictions

After creating your model, click the key link for the model, then click the **Predict** button. 
Select the model to use in the prediction from the drop-down **Model:** menu and the data frame to use in the prediction from the drop-down **Frame:** menu, then click the **Predict** button. 

 ![Making Predictions](images/Flow_makePredict.png)


---
 
<a name="ViewPredict"></a>
## Viewing Predictions

Click the **Assist Me!** button, then click the **getPredictions** link, or enter `getPredictions` in the cell in CS mode and press **Ctrl+Enter**. A list of the stored predictions displays. 
To view a prediction, click the **View** button to the right of the model name. 

 ![Viewing Predictions](images/Flow_getPredict.png)

You can also view predictions by clicking the drop-down **Score** menu and selecting **List All Predictions**.


 

---

<a name="ViewFrame"></a>
## Viewing Frames

To view a specific frame, click the "Key" link for the specified frame, or enter `getFrameSummary "FrameName"` in a cell in CS mode (where `FrameName` is the name of a frame, such as `allyears2k.hex`).

 ![Viewing specified frame](images/Flow_getFrame.png) 


From the `getFrameSummary` cell, you can: 

- view a truncated list of the rows in the data frame by clicking the **View Data** button
- split the dataset by clicking the **Split...** button
- view the columns, data, and factors in more detail or plot a graph by clicking the **Inspect** button
- create a model by clicking the **Build Model** button
- make a prediction based on the data by clicking the **Predict** button
- download the data as a .csv file by clicking the **Download** button
- view the characteristics or domain of a specific column by clicking the **Summary** link

When you view a frame, you can "drill-down" to the necessary level of detail (such as a specific column or row) using the **Inspect** button or by clicking the links. The following screenshot displays the results of clicking the **Inspect** button for a frame.

![Inspecting Frames](images/Flow_inspectFrame.png)

This screenshot displays the results of clicking the **columns** link. 

![Inspecting Columns](images/Flow_inspectCol.png)


To view all frames, click the **Assist Me!** button, then click the **getFrames** link, or enter `getFrames` in the cell in CS mode and press **Ctrl+Enter**. You can also view all current frames by clicking the drop-down **Data** menu and selecting **List All Frames**. 

A list of the current frames in Tetra displays that includes the following information for each frame: 


- Link to the frame (the "key")
- Number of rows and columns
- Size 


For parsed data, the following information displays: 

- Link to the .hex file
- The **Build Model**, **Predict**, and **Inspect** buttons

 ![Parsed Frames](images/Flow_getFrames.png)

To make a prediction, check the checkboxes for the frames you want to use to make the prediction, then click the **Predict on Selected Frames** button. 

---

### Splitting Frames

Datasets can be split within Flow for use in model training and testing. 

 ![splitFrame cell](images/Flow_splitFrame.png)

1. To split a frame, click the **Assist Me** button, then click **splitFrame**.
  
  >**Note**: You can also click the drop-down **Data** menu and select **Split Frame...**.
2. From the drop-down **Frame:** list, select the frame to split. 
3. In the second **Ratio** entry field, specify the fractional value to determine the split. The first **Ratio** field is automatically calculated based on the values entered in the second **Ratio** field. 
   
  >**Note**: Only fractional values between 0 and 1 are supported (for example, enter `.5` to split the frame in half). The total sum of the ratio values must equal one. Tetra automatically adjusts the ratio values to equal one; if unsupported values are entered, an error displays.  
4. In the **Key** entry field, specify a name for the new frame. 
5. (Optional) To add another split, click the **Add a split** link. To remove a split, click the `X` to the right of the **Key** entry field. 
6. Click the **Create** button.  

---
### Creating Frames

To create a frame with a large amount of random data (for example, to use for testing), click the drop-down **Admin** menu, then select **Create Synthetic Frame**. Customize the frame as needed, then click the **Create** button to create the frame. 

  ![Flow - Creating Frames](images/Flow_CreateFrame.png)

---

### Plotting Frames

To create a plot from a frame, click the **Inspect** button, then click the **Plot** button. 

Select the type of plot (point, path, or rect) from the drop-down **Type** menu, then select the x-axis and y-axis from the following options: 

- label 
- type
- missing 
- zeros
- +Inf
- -Inf
- min
- max
- mean
- sigma
- cardinality

Select one of the above options from the drop-down **Color** menu to display the specified data in color, then click the **Plot** button to plot the data. 

 ![Flow - Plotting Frames](images/Flow_plot.png)

>**Note**: Because Tetra stores enums internally as numeric then maps the integers to an array of strings, any `min`, `max`, or `mean` values for categorical columns are not meaningful and should be ignored. Displays for categorical data will be modified in a future version of Tetra. 

---

<a name="Flows"></a>

# ... Using Flows

You can use and modify flows in a variety of ways:

- Clips allow you to save single cells 
- Outlines display summaries of your workflow
- Flows can be saved, duplicated, loaded, or downloaded

---


<a name="Clips"></a>

## Using Clips

Clips enable you to save cells containing your workflow for later reuse. To save a cell as a clip, click the paperclip icon to the right of the cell (highlighted in the red box in the following screenshot). 
 ![Paperclip icon](images/Flow_clips_paperclip.png)

To use a clip in a workflow, click the "Clips" tab in the sidebar on the right. 

 ![Clips tab](images/Flow_clips.png)

All saved clips, including the default system clips (such as `assist`, `importFiles`, and `predict`), are listed. Clips you have created are listed under the "My Clips" heading. To select a clip to insert, click the circular button to the left of the clip name. To delete a clip, click the trashcan icon to right of the clip name. 

>**NOTE**: The default clips listed under "System" cannot be deleted. 

Deleted clips are stored in the trash. To permanently delete all clips in the trash, click the **Empty Trash** button. 

>**NOTE**: Saved data, including flows and clips, are persistent as long as the same IP address is used for the cluster. If a new IP is used, previously saved flows and clips are not available. 

---

<a name="Outline"></a>
## Viewing Outlines

The **Outline** tab in the sidebar displays a brief summary of the cells currently used in your flow; essentially, a command history. 

- To jump to a specific cell, click the cell description. 
- To delete a cell, select it and press the X key on your keyboard. 

 ![View Outline](images/Flow_outline.png)

---

<a name="SaveFlow"></a>
## Saving Flows

You can save your flow for later reuse. To save your flow as a notebook, click the "Save" button (the first button in the row of buttons below the flow name), or click the drop-down "Flow" menu and select "Save Flow." 
To enter a custom name for the flow, click the default flow name ("Untitled Flow") and type the desired flow name. A pencil icon indicates where to enter the desired name. 

 ![Renaming Flows](images/Flow_rename.png)

To confirm the name, click the checkmark to the right of the name field. 
 
 ![Confirm Name](images/Flow_rename2.png)

To reuse a saved flow, click the "Flows" tab in the sidebar, then click the flow name. To delete a saved flow, click the trashcan icon to the right of the flow name. 

 ![Flows](images/Flow_flows.png)

### Finding Saved Flows on your Disk
 
By default, flows are saved to the `h2oflows` directory underneath your home directory.  The directory where flows are saved is printed to stdout:
 
```
03-20 14:54:20.945 172.16.2.39:54323     95667  main      INFO: Flow dir: '/Users/<UserName>/h2oflows'
```

To back up saved flows, copy this directory to your preferred backup location.  

To specify a different location for saved flows, use the command-line argument `-flow_dir` when launching Tetra:

`java -jar h2o.jar -flow_dir /<New>/<Location>/<For>/<Saved>/<Flows>`  

where `/<New>/<Location>/<For>/<Saved>/<Flows>` represents the specified location.  If the directory does not exist, it will be created the first time you save a flow.

### Copying Flows

To create a copy of the current flow, select the **Flow** menu, then click **Make a Copy**. The name of the current flow changes to `Copy of <FlowName>` (where `<FlowName>` is the name of the flow). You can save the duplicated flow using this name by clicking **Flow** > **Save Flow**, or [rename it](#SaveFlow) before saving. 


### Downloading Flows

After saving a flow as a notebook, click the **Flow** menu, then select **Download this Flow**. A new window opens and the saved flow is downloaded to the default downloads folder on your computer. The file is exported as `<filename>.flow`, where `<filename>` is the name specified when the flow was saved. 

**Caution**: You must have an active internet connection to download flows. 

### Loading Flows

To load a saved flow, click the **Flows** tab in the sidebar at the right. In the pop-up confirmation window that appears, select **Load Notebook**, or click **Cancel** to return to the current flow. 

 ![Confirm Replace Flow](images/Flow_confirmreplace.png)

After clicking **Load Notebook**, the saved flow is loaded. 

To load an exported flow, click the **Flow** menu and select **Open Flow...**. In the pop-up window that appears, click the **Choose File** button and select the exported flow, then click the **Open** button. 

 ![Open Flow](images/Flow_Open.png)

>**Notes**: 

>- Only exported flows using the default .flow filetype are supported. Other filetypes will not open. 
- If the current notebook has the same name as the selected file, a pop-up confirmation appears to confirm that the current notebook should be overwritten. 

---

<a name="Troubleshooting"></a>
# ...Troubleshooting Flow

To troubleshoot issues in Flow, use the **Admin** menu. The **Admin** menu allows you to check the status of the cluster, view a timeline of events, and view or download logs for issue analysis. 

>**NOTE**: To view the current Tetra Flow version, click the **Help** menu, then click **About**. 

## Viewing Cluster Status

Click the **Admin** menu, then select **Cluster Status**. A summary of the status of the cluster (also known as a cloud) displays, which includes the same information: 

- Cluster health
- Whether all nodes can communicate (consensus)
- Whether new nodes can join (locked/unlocked)
  
  >**Note**: After you submit a job to Tetra, the cluster does not accept new nodes. 
- Tetra version
- Number of used and available nodes
- When the cluster was created

 ![Cluster Status](images/Flow_CloudStatus.png)


The following information displays for each node:   

- IP address (name)
- Time of last ping
- Number of cores
- Load
- Amount of data (used/total)
- Percentage of cached data
- GC (free/total/max)
- Amount of disk space in GB (free/max)
- Percentage of free disk space 

To view more information, click the **Show Advanced** button. 

---

## Viewing CPU Status (Water Meter)

To view the current CPU usage, click the **Admin** menu, then click **Water Meter (CPU Meter)**. A new window opens, displaying the current CPU use statistics. 

---

## Viewing Logs
To view the logs for troubleshooting, click the **Admin** menu, then click **Inspect Log**. 

 ![Inspect Log](images/Flow_viewLog.png)

To view the logs for a specific node, select it from the drop-down **Select Node** menu. 

---

<a name="DL_Logs"></a>
## Downloading Logs

To download the logs for further analysis, click the **Admin** menu, then click **Download Log**. A new window opens and the logs download to your default download folder. You can close the new window after downloading the logs. Send the logs to [h2ostream](mailto:h2ostream@googlegroups.com) or [file a JIRA ticket](#ReportIssue) for issue resolution. 

---

## Viewing Stack Trace Information

To view the stack trace information, click the **Admin** menu, then click **Stack Trace**. 

 ![Stack Trace](images/Flow_stacktrace.png)

To view the stack trace information for a specific node, select it from the drop-down **Select Node** menu. 

---

## Viewing Network Test Results

To view network test results, click the **Admin** menu, then click **Network Test**. 

  ![Network Test Results](images/Flow_NetworkTest.png)

---

## Accessing the Profiler

The Profiler looks across the cluster to see where the same stack trace occurs, and can be helpful for identifying activity on the current CPU. 
To view the profiler, click the **Admin** menu, then click **Profiler**. 

 ![Profiler](images/Flow_profiler.png)

To view the profiler information for a specific node, select it from the drop-down **Select Node** menu. 

---


## Viewing the Timeline

To view a timeline of events in Flow, click the **Admin** menu, then click **Timeline**. The following information displays for each event: 

- Time of occurrence (HH:MM:SS:MS)
- Number of nanoseconds for duration
- Originator of event ("who")
- I/O type
- Event type
- Number of bytes sent & received

 ![Timeline](images/Flow_timeline.png)

To obtain the most recent information, click the **Refresh** button.  

---

<a name="ReportIssue"></a>
## Reporting Issues

If you experience an error with Flow, you can submit a JIRA ticket to notify our team. 

1. First, click the **Admin** menu, then click **Download Logs**. This will download a file contains information that will help our developers identify the cause of the issue.  
2. Click the **Help** menu, then click **Report an issue**. This will open our JIRA page where you can file your ticket.  
3. Click the **Create** button at the top of the JIRA page. 
4. Attach the log file from the first step, write a description of the error you experienced, then click the **Create** button at the bottom of the page. Our team will work to resolve the issue and you can track the progress of your ticket in JIRA. 

---
