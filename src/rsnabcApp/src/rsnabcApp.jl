module rsnabcApp

import MLUtils

using MLUtils: DataLoader
# using MLUtils: filterobs
# using FastVision: isimagefile
# using FastAI.Datasets: loadfolderdata
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

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--debug_csv"
        help = "Bypass predict and write a dummy submission.csv"
        action = :store_true
        "--batchsize"
        help = "Number of test images to process per batch"
        arg_type = Int
        default = 32
        "--input_size"
        help = "Resize inputs to this. By default aspect ratio is not preserved."
        arg_type = Int
        default = 256
        "--preserve_aspect"
        help = "Aspect ratio must match model input size"
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
    pathdata = loadfolderdata(
        srcdir;
        pattern="*/*")


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
        save(string(dstp) * ".jpg", img_presized) #why not png?
    end
end


function runmodel(input_dir, args)
    taskmodel = args["model"]
    task, model = loadtaskmodel(taskmodel)
    model = gpu(model)
    minibatch = args["batchsize"]
    predictions = []
    # arrayloader = DataLoader(inputs; batchsize=minibatch, shuffle=false)
    arrayloader = loadfolderdata(input_dir, loadfn=loadfile; pattern="*/*")
    # arrayloader = loadfolderdata(dir, filterfn=FastVision.isimagefile, loadfn=(loadfile, parentname))
    for x in 1:length(arrayloader)
        tmpred = FastAI.predict(task, model, getobs(arrayloader, x); device=gpu, context=Inference())
        predictions = push!(predictions, tmpred...)
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
    else
        DSTDIR = Path(mktempdir())
        presizeimagedir(DSTDIR, args)
        # input = loadfolderdata(convert(String, DSTDIR); pattern="*/*")
        preds = runmodel(convert(String, DSTDIR), args)
        tmpdf = DataFrame("cancer" => preds)
        df_pred = hcat(df, tmpdf)
    end
    df_subm = filter_results(df_pred)
    return df_subm
end

function write_submission(df::DataFrame)
    CSV.write("submission.csv", df; bufsize=2^23)
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
