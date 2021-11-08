from invoke import task
import mnist_shootout

@task
def zad1_train(ctx):
    mnist_shootout.zad1_train()

@task
def show_worst_images(ctx, path):
    mnist_shootout.show_worst_images(path)

@task
def zad2(ctx):
    mnist_shootout.zad2()

@task
def zad3(ctx):
    mnist_shootout.zad3()

@task
def zad4(ctx):
    mnist_shootout.zad4()

@task
def zad5(ctx):
    mnist_shootout.zad5()

@task
def zad6(ctx, kernel):
    mnist_shootout.zad6(kernel)
