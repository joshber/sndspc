function kPCA( X; varratio=.99 )
  #=
  X is (n, d): n samples, d dimensions
  https://arxiv.org/abs/1404.1100
  https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca

  Returns a projection of X onto the first k principal directions,
  where those k directions account for varratio of the total variance
  =#

  # Center and normalize X
  n = size(X, 1)
  Xµ = X .- repmat(mean(X, 1), 1, n) # Subtract columnwise means from each row
  Xν = Xµ ./ sqrt(n - 1)

  # FIXME: DO WE NEED TO NORMALIZE? OR JUST CENTER?

  U, Σ, V = svd(Xν)
  @show U
  @show Σ
  @show V

  # Determine how many components we need
  # to preserve the desired proportion of variance
  # FIXME: Do we need to sort Σ in descending order?
  Σd = diagm(Σ)
  var = Σd .* Σd
  totalvar = sum(var[:])
  pvar = 0.0
  k = 0
  while pvar / totalvar < varratio
    k += 1
    pvar += var[k,k]
  end

  # FIXME: projection = U[:,1:k] * Σd[1:k,1:k]
  # Do we need to add means back to decenter?
  # Or get the projection the other way, X * V?

  # FIXME: Is this interpretation correct? Or are principal axes in U, not V? (since Shlens assumes X has samples in columns)
end
