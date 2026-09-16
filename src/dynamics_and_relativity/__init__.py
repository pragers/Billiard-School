import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm


def main() -> None:
    # get data
    data_file ="/Users/jmeij/Downloads/test.txt"
    distance = float(input("Totale afstand(cm):"))


    frame_rate = 120
    data = pl.read_csv(data_file, has_header= False,separator="\t", schema= {"x": pl.Float64, "y":pl.Float64}, row_index_name="frame")
    pixel_distance = ((data.get_column("x").item(0) - data.get_column("x").item(-1))**2 +(data.get_column("y").item(0) - data.get_column("y").item(-1))**2)**0.5

    #insert columns
    data.insert_column(1, ((pl.col("frame")-1)/frame_rate).alias("time"))
    data.insert_column(data.shape[1],(pl.col("x").diff()).alias("x_change"))
    data.insert_column(data.shape[1],(pl.col("y").diff()).alias("y_change"))
    data.insert_column(data.shape[1],(np.sqrt(pl.col("x_change")**2 +pl.col("y_change")**2)).alias("velocity(px/s)"))
    data.insert_column(data.shape[1],(pl.col("velocity(px/s)")/pixel_distance*(distance/100)).alias("velocity(m/s)"))

    # create linear fit with velocity as dependency of time.
    data = data.drop_nulls()
    time = data["time"].to_numpy()
    vel  = data["velocity(m/s)"].to_numpy()
    Time = sm.add_constant(time)
    model = sm.OLS(vel, Time).fit()


    # save model summary to file.
    summary = model.summary(
        xname=["Intercept (v0)", "Acceleration (m^2/s)"],
        yname="Velocity (m/s)"
    )
    with open("reg_sum.txt", "w") as f:
        f.write(summary.as_txt())


    ## create plot
    fig, ax= plt.subplots()
    ax.scatter(
        x=data["time"],
        y=data["velocity(m/s)"],
    )
    ax.set(xlabel = "T (s)", ylabel="V (m/s)")
    ax.errorbar(data["time"],data["velocity(m/s)"],yerr=0.1, capsize=5, fmt="ok", label = "data")
    ax.grid()
    ax.set_title("Ball speed")
    fig.savefig("./plot.svg")

    ##write to output
    data.write_csv("./output.csv", separator = ",")
