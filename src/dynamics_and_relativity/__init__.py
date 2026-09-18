import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm


def main() -> None:
    #loop through all 10 measurements
    for i in range(1,11):
        # get data
        data_file =f"/path/to/Results meting {i}.csv"
        output_folder = f"./results/meting{i}/"
        #distance = float(input("Totale afstand(cm):"))
        os.makedirs(output_folder, exist_ok=True)

        frame_rate = 400
        raw_data= pl.read_csv(data_file, has_header=True, separator=",", row_index_name= "frame")
        data= raw_data.select(["frame","X","Y"])

        #data = pl.read_csv(data_file, has_header= False,separator="\t", schema= {"x": pl.Float64, "y":pl.Float64}, row_index_name="frame")
        pixel_distance = ((data.get_column("X").item(0) - data.get_column("X").item(-1))**2 +(data.get_column("Y").item(0) - data.get_column("Y").item(-1))**2)**0.5

        #insert columns
        data.insert_column(1, ((pl.col("frame")-1)/frame_rate).alias("time"))
        data.insert_column(data.shape[1],(pl.col("X").diff()).alias("x_change"))
        data.insert_column(data.shape[1],(pl.col("Y").diff()).alias("y_change"))
        data.insert_column(data.shape[1],(np.sqrt(pl.col("x_change")**2 +pl.col("y_change")**2)*frame_rate).alias("velocity(cm/s)"))
        data.insert_column(data.shape[1],(pl.col("velocity(cm/s)")/100).alias("velocity(m/s)"))

        # create linear fit with velocity as dependency of time.
        data = data.drop_nulls()
        time = data["time"].to_numpy()
        vel  = data["X"].to_numpy()
        Time = sm.add_constant(time)
        model = sm.OLS(vel, Time).fit()


        # save model summary to file.
        summary = model.summary(
            xname=["Intercept (x0)", "velocity (cm/s)"],
            yname="X"
        )
        with open(output_folder+"reg_sum.txt", "w") as f:
            f.write(summary.as_text())


        ## create plot
        fig, ax= plt.subplots(nrows=2)
        ax[0].scatter(
            x=data["time"],
            y=data["velocity(cm/s)"],
        )
        ax[0].set(xlabel = "T (s)", ylabel="V (cm/s)")
        ax[0].grid()
        ax[0].set_title("Ball speed")
        ax[1].scatter(x=data["time"],y=data["X"])
        ax[1].set(xlabel = "T (s)", ylabel="X (cm)")
        ax[1].grid()
        ax[1].set_title("Ball place")
        plt.tight_layout()
        fig.savefig(output_folder+"plot.svg")

        ##write to output
        data.write_csv(output_folder+"/output.csv", separator = ",")
