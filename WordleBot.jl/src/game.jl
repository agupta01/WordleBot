using Base
using Distributions
using Pipe


mutable struct WordleGame
    solved::Bool
    currstep::Int
    verbose::Int
    score::Float16
    targetword::Union{String,Nothing}
    wordlist::Vector{String}
    accuracylist::Vector{Any}

    function WordleGame(; solved=0,
                        currstep=0,
                        verbose=0,
                        score=10.0,
                        targetword=nothing,
                        wordlist=String[],
                        accuracylist=[])
        words::Vector{String} = @pipe "../words.txt" |> open(f->read(f, String), _) |> split(_, "\n")
        if targetword === nothing
            targetword = words[rand(1:length(words))]
        end
        return new(solved, currstep, verbose, score, targetword, wordlist, accuracylist)
    end
end

"Play one turn."
function turn!(game::WordleGame, guess::String)::Vector{Float16}
    if length(guess) != 5
        error("Guess must be 5 letters long")
    end
    if game.solved
        error("Game is already solved")
    end

    game.currstep += 1
    push!(game.wordlist, guess)

    result = Array{Union{Float16, Missing}}(missing, 5)
    for i in 1:5
        result[i] = game.targetword[i] == guess[i] ? 1.0 : guess[i] in game.targetword ? 0.5 : 0.0
    end
    
    game.score -= 5 - sum(result)

    getsquares(i) = i == 1 ? "🟩" : i == 0.5 ? "🟨" : "⬛"
    if game.verbose == 2
        println(guess * " => " * join(map(getsquares, result)))
    end

    if guess == game.targetword
        game.solved = true
        if game.verbose >= 1
            println("Wordle solved in $(game.currstep) steps. Score: $(game.score)")
        end
    end

    push!(game.accuracylist, result)
    return result
end

"Reset a game."
function reset!(game::WordleGame)::Nothing
    game.solved = false
    game.currstep = 0
    game.score = 10.0
    game.wordlist = String[]
    game.accuracylist = []
    return nothing
end
    

function main()
    w = WordleGame()
end

main()

export WordleGame
