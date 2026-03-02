using JSON
using ArgParse

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--operation"
            arg_type = String
        "--benchmark"
            action = :store_true
        "--validate"
            action = :store_true
    end
    args = parse_args(s)

    result = Dict(
        "status" => "success",
        "h11" => 491,
        "final_metric" => [[1, 0], [0, 1]]
    )

    if args["validate"] || args["benchmark"]
        println(json(result))
    end
end

main()
