
##############################
## Misc
##############################

# Generate unique names for saving models
function runfileid()
    time = now()
    name = replace("$time", ":" => ".")
    return name
end


##############################
## FastVision.jl image stats
##############################
function imagestats(img::AbstractArray{T,N}, C) where {T,N}
    imt = DataAugmentation.imagetotensor(map(x -> convert(C, x), img))
    means = reshape(mean(imt; dims = 1:N), :)
    stds = reshape(std(imt; dims = 1:N), :)
    return means, stds
end


function imagedatasetstats(data, C; progress = true)
    means, stds = imagestats(getobs(data, 1), C)
    loaderfn = d -> eachobs(d, parallel = true, buffer = false)

    p = Progress(numobs(data), enabled = progress)
    for (means_, stds_) in mapobs(img -> imagestats(img, C), data) |> loaderfn
        means .+= means_
        stds .+= stds_
        next!(p)
    end
    return means ./ numobs(data), stds ./ numobs(data)
end

### example usage
#means, stds = imagedatasetstats(images, Gray{N0f8}; progress = true)


##############################
## Data locations
##############################
function choosedata()
    dataset = Dict(
        "pngs" => ("exp_pro", "images_as_pngs", "train_images_processed"),
        "pngs_512" => ("exp_pro", "images_as_pngs_512", "train_images_processed"),
        "pngs_768" => ("exp_pro", "images_as_pngs_768", "train_images_processed"),
        "pngs_1024" => ("exp_pro", "images_as_pngs_1024", "train_images_processed"),
    )
    return dataset
end

# nfsdatadir(args...) = projectdir("../", "data", "HuBMAP", "data", args...)
# traindir(argz...) = nfsdatadir(datasets[args.dataset][1]..., argz...)
