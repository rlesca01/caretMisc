% Generated by roxygen2 (4.1.0): do not edit by hand
% Please edit documentation in R/funcs.r
\name{CatClassML}
\alias{CatClassML}
\title{Wrapper for performing machine learning.}
\usage{
CatClassML(data.mod, mods, mode = NULL, parallel = FALSE, method = "cv",
  number = 10, repeats = 3, verboseIter = TRUE, savePredictions = TRUE,
  tuneLength = 5, fixedIndx = TRUE, grid.list = NULL)
}
\arguments{
\item{data.mod}{is a data.frame with pred as the last column}

\item{mods}{a character vector of valid models to use}

\item{mode}{(optional) either regress or class}

\item{parallel}{either TRUE or FALSE}

\item{fixedIndx}{TRUE or FALSE, set to true for ensemble building}

\item{grid.list}{a list of data.frames specifing custom grid params.  Should be made what the model is named.}
}
\value{
A list of models equal to the number of models (mods) specified.
}
\description{
\code{CatClassML} returns models that were cross validated using specified method.
}

