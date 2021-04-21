## Build instruction for Ubuntu 20.04

0. Install Java 11, NPM

1. Create python environment
```bash
conda create -n h2o python=3.7
```

2. Download and install R
```bash
sudo apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/' 
sudo apt install r-base build-essential
```

3. Install R packages
```
# sudo R
> install.packages("RCurl")
> install.packages("jsonlite")
> install.packages("statmod")
> install.packages("devtools")
> install.packages("roxygen2")
> install.packages("testthat")
```

4. Install Python packages
```bash
cnoda activate h2o
pip install grip colorama future tabulate requests wheel
```

5. Fresh cloned build
```bash
git clone http://...
./gradlew syncSmalldata (optional)
./gradlew syncRPackages (optional)
./gradlew build -x test (-x test means no tests)
```
