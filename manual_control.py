import pygame
import numpy as np
from scipy.integrate import ode
from model import bicycle

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Car Simulation Game")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Car parameters
car_length = 20
car_width = 10

# Game parameters
acceleration = 0
omega = 0

# Initial state [x, y, v, theta]
state = np.array([width/2, height/2, 0, 0])

# Set up ODE solver
solver = ode(bicycle)
solver.set_integrator('dopri5')
solver.set_initial_value(state, 0)

# Game loop
running = True
clock = pygame.time.Clock()

prev_display_time = pygame.time.get_ticks()
while running:
    dt = clock.tick(60) / 1000.0  # Time step in seconds
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Handle continuous key presses
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        acceleration += 1
    if keys[pygame.K_s]:
        acceleration -= 1
    if keys[pygame.K_q]:
        omega += 0.1
    if keys[pygame.K_e]:
        omega -= 0.1
    acceleration = np.maximum(0, np.minimum(5, acceleration))
    omega = np.maximum(-1, np.minimum(1, omega))
    # Apply some decay to acceleration and omega
    acceleration *= 0.95
    omega *= 0.95

    # Solve ODE
    solver.set_f_params(np.array([acceleration, omega]))
    state = solver.integrate(solver.t + dt)

    # Clear the screen
    screen.fill(WHITE)

    if(pygame.time.get_ticks() - prev_display_time > 1000):
        prev_display_time = pygame.time.get_ticks()
        print("State: ",state)
        print("Acceleration: ",acceleration)
        print("omega: ",omega)
    # Draw the car
    x, y, _, theta = state
    y = -y + height
    theta = -theta
    car_points = [
        (x + car_length/2 * np.cos(theta) - car_width/2 * np.sin(theta),
         y + car_length/2 * np.sin(theta) + car_width/2 * np.cos(theta)),
        (x + car_length/2 * np.cos(theta) + car_width/2 * np.sin(theta),
         y + car_length/2 * np.sin(theta) - car_width/2 * np.cos(theta)),
        (x - car_length/2 * np.cos(theta) + car_width/2 * np.sin(theta),
         y - car_length/2 * np.sin(theta) - car_width/2 * np.cos(theta)),
        (x - car_length/2 * np.cos(theta) - car_width/2 * np.sin(theta),
         y - car_length/2 * np.sin(theta) + car_width/2 * np.cos(theta))
    ]
    pygame.draw.polygon(screen, RED, car_points)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()