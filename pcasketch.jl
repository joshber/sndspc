function standardize( X )
  #=
  X is (n, d): n samples, d dimensions
  Returns:
  * X translated and scaled to N(0, 1)
  * n
  =#
  n = size(X,1)
  ( X - repmat(mean(X, 1), n) ) ./ repmat(std(X, 1), n), n
end

function embed( X; minvar=.99 )
  #=
  X is (n, d): n samples, d dimensions

  Returns:
  * A projection of standardized X onto the first k principal directions,
    where those k directions account for varratio of the total variance
  * The first k principal directions as a matrix, with PDs in columns
    (projection basis)

  Principal component analysis is one of those things that makes less sense
  the more you think about it. Confusion abounds: Samples in rows or in columns,
  decompose the sample matrix or the covariance matrix, choice of decomposition
  (eig vs svd), efficiency and numerical stability of different strategies.
  I've found these helpful:

  https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
  https://arxiv.org/abs/1404.1100
  https://stats.stackexchange.com/questions/79043/why-pca-of-data-by-means-of-svd-of-the-data

  Strategy:
  * Standardize X
  * Get correlation matrix (or kernel-based similarity matrix)
  * Decompose with SVD
  * Eigenvalues = diagm(Σ) — Sum to find # directions to meet variance threshold
  * Projection = XV (standardized X)
  =#

  Xs, n = standardize(X)

  sim = Xs'Xs ./ (n - 1) # TODO: This is where we'll swap in the Gaussian kernel

  U, Σ, V = svd(sim) # TODO: svdfact, then extract Σ and V — we don't need U

  # Determine how many principal directions we need
  # in order to preserve the desired proportion of variance
  # Σ comes sorted in descending order
  totalvar = sum(Σ)
  pvar = 0.0
  k = 0
  while pvar / totalvar < minvar
    k += 1
    pvar += var[k]
  end

  Xs * V[:,1:k], V[:,1:k]
end
