using HDF5
using GLVisualize, GeometryTypes, Colors
using Plots


function envelogram( source, X, Y, fps )
  #=
  Draw something like a spectrogram but for projected feature dimensions
  Here X, Y are a (n, d1) and (n, d2) arrays of intensities in projected spaces (e.g., PCA, LTSA)

  Inspiration: http://www.glvisualize.com/examples/imagelike/#contourf
  TODO: Perspective camera, ribbons of lines as http://www.glvisualize.com/examples/lines/#lines3d
  =#

  Xmn = minimum(X)
  Xmx = maximum(X)
  Ymn = minimum(Y)
  Ymx = maximum(Y)

  # Translate and scale
  Xscaled = [ Float32((i - Xmn) / (Xmx - Xmn)) for i in X ]
  Yscaled = [ Float32((i - Ymn) / (Ymx - Ymn)) for i in Y ]

  # Combine the two projections for visualization
  scaled = hcat(Xscaled, zeros(Float32, size(X,1), 1), Yscaled)

  barw = 2 #10 # bar width in px
  gapw = 2
  framelen = 30 * fps # 30s visible frame

  # We could use Simplex or HyperRectangles as well as columns of intensities
  # It's a question of how _default() dispatches — see http://www.glvisualize.com/api/
  # Achtung, we're transposing from scaled to buf — rows become x coordinates in rendering
  buf = zeros(Intensity{1,Float32}, (barw + gapw) * size(scaled,2), size(scaled,1) + 2 * framelen)
  framebuf = zeros(Intensity{1,Float32}, size(buf,1), framelen)

  # buf gets a frame's worth of zeros at either end — simplifies buffer copying
  for i in 1:size(scaled,2)
    buf[(barw + gapw) * (i - 1) + 1 : (barw + gapw) * (i - 1) + barw, framelen + 1 : framelen + size(scaled,1)] .=
      Intensity{1,Float32}.(scaled[:,i]')
  end

  window = glscreen("$(source)", resolution=(2 * size(buf,1), framelen))
  timesignal = loop(1 : size(buf,2) - framelen, fps)

  renderable = visualize(map( t -> framebuf .= buf[:, t : t + framelen - 1], timesignal))

  _view(renderable, window, camera=:orthographic_pixel)

  renderloop(window)
end

function heat( source, X, fps )
  #=
  Plot projected feature space as a series of histogram sparklines
  https://juliaplots.github.io/examples/plotlyjs/
  https://juliaplots.github.io/layouts/
  =#
  # Normalize data? Do I even need to?
  # Use histogram() or bar() — bar allows negative values
  # How to remove whitespace between bars?
  # Do I even need fps? To show total length?
  # Construct layout — dimensions stacked vertically
  # Or just pass plot() layout attribute tuple: layout=(d,1)
  # plot attributes: t=:bar (or simply use bar()) l=(n,1) leg=false, ticks=nothing, border=false, palette=…

  plotlyjs()

  #=mn = minimum(X)
  mx = maximum(X)
  X = [ Float32((i - mn) / (mx - mn)) for i in X ]
  =#

  #plot(X; t=:line, layout=(size(X,2), 1), axis=:none, key=false, ticks=nothing, border=false)
  heatmap(1:size(X,1), 1:size(X,2), X')
  gui()
  #png("$(source)")
  #l = @layout([a{0.1h};b [c;d e]])
  #plot(randn(100,5),layout=l,t=[:line :histogram :scatter :steppre :bar],leg=false,ticks=nothing,border=false)
end

source = "小学校/170902_01"
path = "/Users/josh/Dropbox/Shirooni/Recordings/out/$(source).h5"

f = h5open(path)
data = read(f)
close(f)

fps = data["fps"]
X = data["X"]
P1 = data["pca"]
P2 = data["ltsa"]

heat("$(source)", X, fps)

#envelogram("$(source)", X, P1, fps)
