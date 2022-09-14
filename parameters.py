import numpy
import datetime

# define batch size, learning rate and number of steps here for this project
def basic_parameters():
    batch_size = numpy.array([4])
    learning_rate = numpy.array([0.0001])
    number_of_steps = [200000]
    return learning_rate, batch_size, number_of_steps


def naming_convention(name, lr, batch_size, noofsteps):
    date = datetime.datetime.now()
    name = name + "-" + date.strftime("%b") + "-" + date.strftime("%d") + "-" \
           + date.strftime("%H") + "-" + \
           date.strftime("%M") + "-" + date.strftime("%S")
    experiment_name = name + "-lr-" + str(lr) + "-batch_size-" + str(batch_size) + \
                      "-no_of_steps-" + str(noofsteps)
    log_name = experiment_name
    saved_model_name = experiment_name

    # Create new checkpoint path
    check_point_name = experiment_name

    return experiment_name, log_name, saved_model_name, check_point_name
