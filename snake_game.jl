module SnakeGame

import Base.size
import Base.print
import Base.rand
import Base.+
import Base.==

@enum Direction begin
           up = 1
           down = 2
           left = 3
           right = 4
       end

mutable struct Position
    x::Int
    y::Int
    Position(x::Int,y::Int) = new(x,y)
end

(+)(p1::Position, p2::Position) = Position(p1.x+p2.x,p1.y+p2.y)
(==)(p1::Position, p2::Position) = p1.x == p2.x && p1.y == p2.y

mutable struct Snake
    positions::Array{Position,1}
end

function Snake(pos::Position)::Snake
    positions = Array{Position,1}(undef,0)
    push!(positions, pos)
    Snake(positions)
end

function size(snake::Snake)::Int
    return size(snake.positions)[1]
end

mutable struct Game
    size::Int
    snake::Snake
    food::Position
    over::Bool
    num_move::Int
end

function Game(size::Int)::Game
    food = Position(rand(1:size),rand(1:size)) # TODO put that back
    #food = Position(1,1)
    snake_pos = trunc(Int, (size/2)+1)
    if food == Position(snake_pos,snake_pos)
        food = Position(1,1)
    end
    Game(size,Snake(Position(snake_pos,snake_pos)),food,false,0)
end

function next_frame!(game::Game,dir::Direction)::Bool
    game.num_move += 1
    x::Int = 0
    y::Int = 0
    if dir == up
        y -= 1
    elseif dir == down
        y += 1
    elseif dir == right
        x += 1
    else
        x -= 1
    end
    snake_head::Position = last(game.snake.positions) + Position(x,y)
    if snake_head != game.food
        deleteat!(game.snake.positions,1) # Cutting tail
    else
        game.food = available_position(game)
    end

    snake_body::Array{Position,1} = game.snake.positions[1:end]
    push!(game.snake.positions,snake_head) # Pushing head
    if snake_head in snake_body || out_of_bound(game,snake_head)
        game.over = true
    end
    return game.over
end

function out_of_bound(game::Game,pos::Position)::Bool
    if pos.x < 1 || pos.y < 1 || pos.x > game.size || pos.y > game.size
        return true
    else
        return false
    end
end

function available_position(game::Game)::Position
    positions = Array{Position,1}(undef,0)
    for y = 1: game.size
        for x = 1: game.size
            pos = Position(x,y)
            if !(pos in game.snake.positions) && pos != game.food
                push!(positions, pos)
            end
        end
    end
    if size(positions)[1] == 0 # Game is full
        return Position(1,1)
    end
    #return Position(1,1)
    return positions[rand(1:size(positions)[1])] # TODO put that back
end

function game_score(game::Game)::Float64
    return 100 * size(game.snake) + game.num_move
    #return game.num_move
    #return 100 * size(game.snake) - game.num_move
    #return size(game.snake) * (game.size^2 - game.num_move)
end

function print(game::Game)
    println("-----SNAKE GAME-----")
    for y = 1: game.size
        for x = 1: game.size
            if Position(x,y) in game.snake.positions
                print("#")
            elseif Position(x,y) == game.food
                print("*")
            else
                print("O")
            end
        end
        print("\n")
    end
end

end
