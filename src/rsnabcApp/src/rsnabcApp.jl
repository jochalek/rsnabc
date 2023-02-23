module rsnabcApp

import MLUtils

using MLUtils: DataLoader, BatchView
using ArgParse
using CSV
using DataFrames
using FastAI
using FastVision
using FileIO
using Images
using FilePathsBase
# using Metalhead
# using DataAugmentation

### Custom FastAI learning task
# include("./tasks.jl")

### Command line arguments
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--debug_csv"
        help = "Bypass predict and write a dummy submission.csv"
        action = :store_true
        "--debug_images"
        help = "Process and save images, bypass predict and write a dummy submission.csv"
        action = :store_true
        "--batchsize"
        help = "Number of test images to process per batch"
        arg_type = Int
        default = 1
        "--input_size"
        help = "Resize inputs to this. By default aspect ratio is not preserved."
        arg_type = Int
        default = 256
        "--preserve_aspect"
        help = "Aspect ratio must match model input size"
        action = :store_true
        "--presize"
        help = "Presize images to disk before running the model."
        action = :store_true
        "--test_data", "-t"
        help = "The directory of the test data including test.csv and test_images/"
        arg_type = String
        default = "/kaggle/input/rsna-breast-cancer-detection"
        "--model", "-m"
        help = "The location of the inference model"
        arg_type = String
        default = "./model.jld2"
    end
    return parse_args(s)
end

"""
Loads file and scales it to the input size. Does not preserve aspect ratio
"""
function grabimage(filepath::String, size)
    image = loadfile(filepath)
    tfm =
        FastVision.DataAugmentation.ScaleFixed((size, size)) |>
        FastVision.DataAugmentation.PinOrigin()
    image = FastVision.DataAugmentation.Image(image)
    timage = FastVision.DataAugmentation.apply(tfm, image)
    return FastVision.DataAugmentation.itemdata(timage)
end
grabimage(filepath::Vector{String}, size) = grabimage.(filepath, size)

#HACK to fix ERROR: MethodError: no method matching loadfile(::Vector{String})
loadfile(file::Vector{String}) = FastAI.loadfile.(file)

function presizeimage(img, args)
    size = args["input_size"]
    if args["preserve_aspect"]
        resized_image = imresize(img, size)
    else
        resized_image = imresize(img, (size, size))
    end
    return resized_image
end

function presizeimagedir(dstdir, args)
    srcdir = joinpath(args["test_data"], "test_images")
    # pathdata = filterobs(isimagefile, loadfolderdata(srcdir)) ## filterobs and pathparents are having a problem? Empty collection...
    pathdata = loadfolderdata(srcdir; pattern = "*/*")


    # create directories beforehand
    for i = 1:numobs(pathdata)
        mkpath(pathparent(getobs(pathdata, i)))
    end

    Threads.@threads for i = 1:numobs(pathdata)
        srcp = getobs(pathdata, i)
        p = relpath(srcp, srcdir)
        dstp = joinpath(dstdir, p)

        img = loadfile(srcp)
        img_presized = presizeimage(img, args)
        save(string(dstp)[1:end-4] * ".jpg", img_presized) #why not png?
    end
end

function runmodel(df::DataFrame, args)
    resized_image = args["input_size"]
    taskmodel = args["model"]
    task, model = loadtaskmodel(taskmodel)
    model = gpu(model)
    minibatch = args["batchsize"]
    predictions = DataFrame()
    paths = [
        joinpath(
            args["test_data"],
            "test_images",
            string(df[row, :patient_id]),
            string(df[row, :image_id]) * ".dcm",
        ) for row = 1:nrow(df)
    ]
    ids = [string(df[row, :prediction_id]) for row = 1:nrow(df)]
    _batchloader = mapobs(paths) do paths
        grabimage(paths, resized_image)
    end
    batchloader = DataLoader((_batchloader, ids); batchsize = minibatch, parallel = true)
    for batch in batchloader
        imgs, pids = batch
        # _df = DataFrame()
        preds = predictbatch(task, model, imgs; device = gpu, context = Inference())
        _df = DataFrame("prediction_id" => pids, "cancer" => preds)
        # _df = hcat(_df, batch)
        # _df = hcat(_df, DataFrame("cancer" => preds))
        predictions = vcat(predictions, _df)
    end
    return predictions
end

function runmodel(input_dir::String, df::DataFrame, args)
    resized_image = args["input_size"]
    taskmodel = args["model"]
    task, model = loadtaskmodel(taskmodel)
    model = gpu(model)
    minibatch = args["batchsize"]
    predictions = DataFrame() #FIXME need to carry :prediction_id from og DataFrame into pipeline to id preds.
    paths = [
        joinpath(
            input_dir,
            string(df[row, :patient_id]) * "_" * string(df[row, :image_id]) * ".png",
        ) for row = 1:nrow(df)
    ]
    ids = [string(df[row, :prediction_id]) for row = 1:nrow(df)]
    _batchloader = mapobs(loadfile, paths)
    batchloader = DataLoader((_batchloader, ids); batchsize = minibatch, parallel = true)
    for batch in batchloader
        imgs, pids = batch
        preds = predictbatch(task, model, imgs; device = gpu, context = Inference())
        _df = DataFrame("prediction_id" => pids, "cancer" => preds)
        predictions = vcat(predictions, _df)
    end
    return predictions
end

function filter_results(df::DataFrame)
    df_subm = DataFrame()
    for id in unique(df[!, :prediction_id])
        piddf = filter(:prediction_id => ==(id), df)
        max = describe(select(piddf, :cancer), :max)[1, 2]
        df_srow = DataFrame("prediction_id" => id, "cancer" => max)
        df_subm = vcat(df_subm, df_srow)
    end
    return df_subm
end


function generate_submission(df::DataFrame, args)
    if args["debug_csv"]
        df_pred = DataFrame()
        for row = 1:nrow(df)
            df_row =
                DataFrame("prediction_id" => df[row, :prediction_id], "cancer" => rand(1))
            df_pred = vcat(df_pred, df_row)
        end
    elseif args["debug_images"]
        DSTDIR = Path(mktempdir())
        presizeimagedir(DSTDIR, args)
        df_pred = DataFrame()
        for row = 1:nrow(df)
            df_row =
                DataFrame("prediction_id" => df[row, :prediction_id], "cancer" => rand(1))
            df_pred = vcat(df_pred, df_row)
        end
    else
        if args["presize"]
            # DSTDIR = Path(mktempdir())
            # presizeimagedir(DSTDIR, args)
            # df_pred = runmodel(convert(String, DSTDIR), df, args)
            DSTDIR = "/tmp/output"
            df_pred = runmodel(DSTDIR, df, args)
        else
            df_pred = runmodel(df, args)
        end
    end
    df_subm = filter_results(df_pred)
    return df_subm
end

function write_submission(df::DataFrame)
    CSV.write("submission.csv", df; bufsize = 2^23)
end

function real_main()
    args = parse_commandline()
    df = DataFrame(CSV.File(joinpath(args["test_data"], "test.csv")))
    df_subm = generate_submission(df, args)
    write_submission(df_subm)
end

function julia_main()::Cint
    # do something based on ARGS?
    try
        real_main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0 # if things finished successfully
end

end # module rsnabcApp
