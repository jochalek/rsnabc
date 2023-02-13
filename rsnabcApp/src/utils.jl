### FastVision.jl image stats
function imagestats(img::AbstractArray{T, N}, C) where {T, N}
    imt = DataAugmentation.imagetotensor(map(x -> convert(C, x), img))
    means = reshape(mean(imt; dims = 1:N), :)
    stds = reshape(std(imt; dims = 1:N), :)
    return means, stds
end


function imagedatasetstats(data,
                           C;
                           progress = true)
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

#means, stds = imagedatasetstats(images, Gray{N0f8}; progress = true)
