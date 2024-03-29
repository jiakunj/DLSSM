% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DLSSM.R
\name{Batched}
\alias{Batched}
\title{Combine data into Batched data}
\usage{
Batched(formula, data, time, S)
}
\arguments{
\item{formula}{an object of class "formula" (or one that can be coerced to that class): a symbolic description of response and covariates in the model.}

\item{data}{The dataset matrix containing the observations (one row per sample).}

\item{time}{The name of the variable in the dataset which represents time. The varying coefficient functions are assumed to be smooth functions of this variable.}

\item{S}{number of batches}
}
\value{
\tabular{ll}{
   \code{batched} \tab List of batched data, the element of list is matrix with each row per sample \cr
   \tab \cr
   \code{gap.len} \tab interval length 1/S  \cr
 }
}
\description{
The time domain of observation will first be standardized into [0,1]. Then [0,1] will be divided into S equally spaced intervals as described in Jiakun et al.(2021, Biometrics).
Then those intervals slice the dataset to S batches of data.
}
\author{
Jiakun Jiang, Wei Yang, Stephen E. Kimmel and Wensheng Guo
}
