# FIXME using DrWatson before activating the environment breaks my FileIO "hack" to load dicom
# using DrWatson
# @quickactivate "rsnabc"
using DrWatson
using rsnabc

using ArgParse,
    CSV,
    DataFrames,
    FastAI,
    FastVision,
    FileIO,
    FilePathsBase,
    Images,
    MLUtils,
    MLDatasets,
    DataAugmentation,
    Random
using Wandb, Logging, Statistics, Metalhead
import FastVision: Gray, N0f8, SVector

println("""
        Currently active project is: $(projectname())

        Path of active project: $(projectdir())

        Have fun with your new project!

        You can help us improve DrWatson by opening
        issues on GitHub, submitting feature requests,
        or even opening your own Pull Requests!
        """)

s = ArgParseSettings()
@add_arg_table! s begin
    "--lr"
    arg_type = Float64
    default = 0.033
    "--epochs"
    arg_type = Int
    default = 1
    "--imgsize"
    arg_type = Int
    default = 128
    "--batchsize"
    arg_type = Int
    default = 2
    "--rng"
    arg_type = Int
    default = 42
    "--backbone"
    arg_type = String
    default = "resnet18"
    "--dataset"
    arg_type = String
    default = "pngs"
    "--prototype"
    action = :store_true
    "--nowandb"
    action = :store_true
end
args = dict2ntuple(parse_args(s))

rng = Random.default_rng()
Random.seed!(rng, args.rng)

## Dataset choice
dataset = choosedata()
traindir(argz...) = datadir(dataset[args.dataset]..., argz...)
modeldir(argz...) = datadir("models", argz...)

## Logging
runname = rsnabc.runfileid()

lgbackend = WandbBackend(
    project = "rsnabc",
    name = "$runname",
    config = Dict(
        "learning_rate" => args.lr,
        "Projective Transforms" => (args.imgsize, args.imgsize),
        "batchsize" => args.batchsize,
        "epochs" => args.epochs,
        "prototype" => args.prototype,
        "backbone" => args.backbone,
        "dataset" => args.dataset,
    ),
)

### Data Prep
# Labels
df = DataFrame(CSV.File(datadir("exp_pro", "train.csv")))
labels = select(df, :cancer)[:, 1];

# Images FIXME which way?
# images = FileDataset(traindir(), "*/*");

images = loadfolderdata(traindir(), filterfn = rsnabc.isimagefile, loadfn = loadfile)

alldata = (images, labels);

### FIXME use taskdataloaders or something to split this before balancing to avoid leakage
train, test = splitobs(alldata, at = 0.7, shuffle = true)
balanced_train = oversample(train[1], train[2]; fraction = 0.34, shuffle = true)

### Task setup
task = BlockTask(
    (FastVision.Image{2}(), Label([0, 1])),
    (
        ProjectiveTransforms((args.imgsize, args.imgsize)),
        ImagePreprocessing(; means = RSNABC_MEANS, stds = RSNABC_STDS, C = Gray{N0f8}),
        OneHot(),
    ),
)

### For tests
# sample = getobs(data, 1)
# x, y = encodesample(task, Training(), sample)
# summary(x), summary(y)


### Metrics
function fscore(yhat, y)
    return 1 - FastAI.Flux.dice_coeff_loss(yhat, y)
end

### Train
f1 = Metric(fscore, device = gpu)
backbone = Metalhead.ResNet(18).layers[1:end-1]
lossfn = FastAI.Flux.logitcrossentropy
learner = tasklearner(
    task,
    balanced_train;
    lossfn = lossfn,
    callbacks = [ToGPU(), Metrics(accuracy, f1)],
)
epochs = args.epochs
lr = args.lr
fitonecycle!(learner, epochs, lr)

### Save the model
savetaskmodel(modeldir(String(runfileid * ".jld2")), task, learner.model, force = true)




### Generic training implementation

# backbone = Metalhead.ResNet(18).layers[1:end-1]
# learner = tasklearner(task, data; callbacks=[ToGPU(), Metrics(accuracy)])
# fitonecycle!(learner, 1, 0.001)
