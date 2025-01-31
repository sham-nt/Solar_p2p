import pygame
import numpy as np
from env import SolarP2PEnv

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Solar P2P Energy Trading Simulation")

# Colors
BACKGROUND = (230, 230, 230)
SOLAR_OWNER = (255, 215, 0)  # Gold
NON_SOLAR_OWNER = (100, 149, 237)  # Cornflower Blue
GRID = (50, 205, 50)  # Lime Green
TEXT_COLOR = (0, 0, 0)
GRID_COLOR = (200, 200, 200)
SOLAR_ENERGY_FLOW = (255, 165, 0)  # Orange
GRID_ENERGY_FLOW = (0, 191, 255)  # Deep Sky Blue

# Fonts
font = pygame.font.Font(None, 24)
title_font = pygame.font.Font(None, 36)

# Load and scale images
solar_image = pygame.image.load("images/solar_panel.png")
solar_image = pygame.transform.scale(solar_image, (80, 80))
house_image = pygame.image.load("images/house.png")
house_image = pygame.transform.scale(house_image, (60, 60))
grid_image = pygame.image.load("images/power_grid.png")
grid_image = pygame.transform.scale(grid_image, (100, 100))

def draw_agent(screen, x, y, image, production, consumption, price, battery=None, name=None):
    # Draw energy bars
    if production is not None:
        pygame.draw.rect(screen, (0, 255, 0), (x - 30, y - 50, 60 * production, 5))
    pygame.draw.rect(screen, (255, 0, 0), (x - 30, y - 40, 60 * consumption, 5))
    if battery is not None:
        pygame.draw.rect(screen, (0, 0, 255), (x - 30, y - 60, 60 * battery, 5))
    
    # Draw agent image
    screen.blit(image, (x - image.get_width() // 2, y - image.get_height() // 2))
    
    # Draw price
    price_text = font.render(f"${price:.2f}", True, TEXT_COLOR)
    screen.blit(price_text, (x - 30, y + 40))

    # Draw name
    if name:
        name_text = font.render(name, True, TEXT_COLOR)
        screen.blit(name_text, (x - 50, y + 60))

def draw_grid(screen, width, height, cell_size):
    for x in range(0, width, cell_size):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, height))
    for y in range(0, height, cell_size):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (width, y))

def draw_energy_flow(screen, start, end, amount, color):
    if amount > 0:
        pygame.draw.line(screen, color, start, end, 2)
        # Draw arrow
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        pygame.draw.polygon(screen, color, [
            (end[0] - 10 * np.cos(angle) - 5 * np.sin(angle),
             end[1] - 10 * np.sin(angle) + 5 * np.cos(angle)),
            (end[0] - 10 * np.cos(angle) + 5 * np.sin(angle),
             end[1] - 10 * np.sin(angle) - 5 * np.cos(angle)),
            end
        ])

def draw_explanation(screen, env, time, scenario):
    explanation = [
        "Solar P2P Energy Trading Simulation",
        "",
        f"Time: {int(time):02d}:00",
        f"Scenario: {scenario}",
        "",
        "Solar Owners (with panels):",
        "- Green bar: Energy production",
        "- Red bar: Energy consumption",
        "- Blue bar: Battery storage",
        "",
        "Non-Solar Owners (houses):",
        "- Red bar: Energy consumption",
        "",
        "Grid (power lines):",
        "- Acts as a backup energy source/sink",
        "",
        "Prices shown below each agent",
        "Orange lines: Solar energy flow",
        "Blue lines: Grid energy flow",
        "",
        "The simulation updates every hour,",
        "showing how energy production,",
        "consumption, and prices change",
        "throughout the day."
    ]

    y_offset = 20
    for line in explanation:
        text = font.render(line, True, TEXT_COLOR)
        screen.blit(text, (WIDTH - 300, y_offset))
        y_offset += 30

def create_button(text, x, y, width, height):
    button = pygame.Rect(x, y, width, height)
    return button

def draw_button(screen, button, text):
    pygame.draw.rect(screen, (200, 200, 200), button)
    text_surf = font.render(text, True, (0, 0, 0))
    text_rect = text_surf.get_rect(center=button.center)
    screen.blit(text_surf, text_rect)

def draw_counters(screen, solar_energy_used, grid_energy_used):
    solar_text = font.render(f"Solar Energy Used: {solar_energy_used:.2f}", True, TEXT_COLOR)
    grid_text = font.render(f"Grid Energy Used: {grid_energy_used:.2f}", True, TEXT_COLOR)
    screen.blit(solar_text, (20, 20))
    screen.blit(grid_text, (20, 50))

def visualize_environment(env):
    running = True
    clock = pygame.time.Clock()

    normal_button = create_button("Normal", 20, HEIGHT - 150, 100, 40)
    low_power_button = create_button("Low Power", 130, HEIGHT - 150, 100, 40)
    high_power_button = create_button("High Power", 240, HEIGHT - 150, 100, 40)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if normal_button.collidepoint(event.pos):
                    env.set_scenario('normal')
                elif low_power_button.collidepoint(event.pos):
                    env.set_scenario('low_power')
                elif high_power_button.collidepoint(event.pos):
                    env.set_scenario('high_power')

        screen.fill(BACKGROUND)
        draw_grid(screen, WIDTH - 320, HEIGHT, 50)

        # Take a random action
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        energy_balance = info["energy_balance"]
        solar_energy_used = info["solar_energy_used"]
        grid_energy_used = info["grid_energy_used"]

        # Draw energy flow lines
        solar_positions = []
        non_solar_positions = []
        for i in range(env.num_solar_owners):
            x = 100 + (i * 180)
            y = 200
            solar_positions.append((x, y))
        for i in range(env.num_non_solar_owners):
            x = 100 + (i * 180)
            y = 450
            non_solar_positions.append((x, y))
        grid_position = (400, 650)

        for i, balance in enumerate(energy_balance[:-1]):  # Exclude grid from this loop
            if i < env.num_solar_owners:
                start = solar_positions[i]
            else:
                start = non_solar_positions[i - env.num_solar_owners]
            
            if balance > 0:  # Selling energy
                for j in range(env.num_solar_owners, env.num_agents - 1):  # To non-solar and grid
                    if j < env.num_solar_owners + env.num_non_solar_owners:
                        end = non_solar_positions[j - env.num_solar_owners]
                    else:
                        end = grid_position
                    draw_energy_flow(screen, start, end, balance, SOLAR_ENERGY_FLOW)
            elif balance < 0:  # Buying energy
                for j in range(env.num_solar_owners):  # From solar
                    draw_energy_flow(screen, solar_positions[j], start, -balance, SOLAR_ENERGY_FLOW)
                draw_energy_flow(screen, grid_position, start, -balance, GRID_ENERGY_FLOW)  # From grid

        # Draw agents
        for i in range(env.num_solar_owners):
            x, y = solar_positions[i]
            draw_agent(screen, x, y, solar_image, state[i, 0], state[i, 1], state[i, 2], state[i, 3], f"Solar {i+1}")

        for i in range(env.num_non_solar_owners):
            x, y = non_solar_positions[i]
            draw_agent(screen, x, y, house_image, None, state[env.num_solar_owners + i, 1], state[env.num_solar_owners + i, 2], name=f"Non-Solar {i+1}")

        draw_agent(screen, *grid_position, grid_image, None, state[-1, 1], state[-1, 2], name="Grid")

        # Draw explanation
        draw_explanation(screen, env, state[0, 4], env.scenario)

        # Draw counters
        draw_counters(screen, solar_energy_used, grid_energy_used)

        # Draw buttons
        draw_button(screen, normal_button, "Normal")
        draw_button(screen, low_power_button, "Low Power")
        draw_button(screen, high_power_button, "High Power")

        pygame.display.flip()
        clock.tick(1)  # Update once per second

        if done:
            env.reset()

    pygame.quit()

# Run the visualization
env = SolarP2PEnv(num_solar_owners=5, num_non_solar_owners=5)
visualize_environment(env)