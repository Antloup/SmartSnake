include("./snake_game.jl")

using ..SnakeGame
# A small script so we can enjoy our snake game :)

function main()
    game_size::Int = 7
    game = SnakeGame.Game(game_size)
    while true
        print(game)
        input::String = readline(stdin)
        if input == "z"
            SnakeGame.next_frame!(game,SnakeGame.up)
        elseif input == "s"
            SnakeGame.next_frame!(game,SnakeGame.down)
        elseif input == "q"
            SnakeGame.next_frame!(game,SnakeGame.left)
        elseif input == "d"
            SnakeGame.next_frame!(game,SnakeGame.right)
        else
            break
        end
        if game.over
            break
        end
    end
    println("Game over")
    println("Score : ",SnakeGame.game_score(game))
end

main()
