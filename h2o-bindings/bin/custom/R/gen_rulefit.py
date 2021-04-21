extensions = dict(
    extensions = dict(
        validate_params="""
# Required maps for different names params, including deprecated params
.gbm.map <- c("x" = "ignored_columns",
              "y" = "response_column")
"""
    ),
    set_required_params="""
parms$training_frame <- training_frame
args <- .verify_dataxy(training_frame, x, y)
parms$ignored_columns <- args$x_ignore
parms$response_column <- args$y
""",
)

doc = dict(
    preamble="""
Build a RuleFit Model
    
Builds a Distributed RuleFit model on a parsed dataset, for regression or 
classification.
    """,
    params=dict(
        model_type="Specifies type of base learners in the ensemble. Must be one of: \"rules_and_linear\", \"rules\", \"linear\". "
                   "Defaults to rules_and_linear.",
        min_rule_length="Minimum length of rules. Defaults to 1.",
        max_rule_length="Maximum length of rules. Defaults to 10."
    ),
    signatures=dict(
        model_type="c(\"rules_and_linear\", \"rules\", \"linear\")"
    ),
    examples="""
library(h2o)
h2o.init()

f <- "https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv"
titanic <- h2o.importFile(f)
response = "survived"
predictors <- c("age", "sibsp", "parch", "fare", "sex", "pclass")
titanic[,response] <- as.factor(titanic[,response])
titanic[,"pclass"] <- as.factor(titanic[,"pclass"])
rfit <- h2o.rulefit(y = response, x = predictors, training_frame = titanic, max_rule_length = 10, 
max_num_rules = 100, seed = 1, model_type = "rules")
"""
)
