module rsnabc

using Dates: now
using ArgParse,
    CSV,
    DataFrames,
    FastVision,
    FileIO,
    FilePathsBase,
    Images,
    MLUtils,
    MLDatasets,
    DataAugmentation,
    Wandb,
    Logging,
    Statistics,
    Metalhead
import ProgressMeter: Progress, next!
import FastVision: Gray, N0f8, SVector

include("utils.jl")

const RE_IMAGEFILE = r".*\.(gif|jpe?g|tiff?|png|webp|bmp|dcm)$"i
isimagefile(f) = matches(RE_IMAGEFILE, f)

# color statistics for normalization
const RSNABC_MEANS = SVector{1}(Float32[0.13926385])
const RSNABC_STDS = SVector{1}(Float32[0.19799942])


export runfileid, choosedata, isimagefile, RSNABC_MEANS, RSNABC_STDS

end # module
