# Optional formula check against the CRAN EValue package.
# This file does not install packages or contact a remote repository.
if (!requireNamespace("EValue", quietly = TRUE)) {
  stop("Install the EValue package before running this optional check.")
}
ratios <- c(1, 0.8, 1.25, 2, 3.9)
formula <- function(rr) {
  magnitude <- max(rr, 1 / rr)
  magnitude + sqrt(magnitude * (magnitude - 1))
}
reference <- vapply(ratios, function(rr) as.numeric(EValue::evalues.RR(rr)[2, 1]), numeric(1))
stopifnot(max(abs(reference - vapply(ratios, formula, numeric(1)))) < 1e-10)
print(data.frame(risk_ratio = ratios, E_value_point = reference))
