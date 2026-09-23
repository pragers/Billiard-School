import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm

from dynamics_and_relativity.err_prop import Formula


def main() -> None:
    #loop through all 10 measurements

    for i in os.scandir("/Users/jmeij/Downloads/OneDrive_1_9-23-2026-2"):
        if not i.is_file or i.name == ".DS_Store":
            continue
        print(i.path)
        # get data
        data_file = i.path
        name_split = os.path.splitext(i.name)
        name = name_split[0]
        output_folder = f"./results week 2/meting {name}/"
        os.makedirs(output_folder, exist_ok=True)

        frame_rate = 1200 # in Hz
        frames_skipped = 4
        shutter_speed = 33e-6 #1/1250 # in s
        delim = "\t"
        if "csv" in name_split[1]:
            delim = ","
        raw_data= pl.read_csv(data_file, has_header=True, separator=delim, row_index_name= "frame")
        data= raw_data.select(["frame","X","Y"])
        formula = Formula("x_change / time_change",vars = ["x_change","time_change"], consts={})

        #insert columns
        data.insert_column(1, ((pl.col("frame"))/frame_rate).alias("time"))
        data.insert_column(data.shape[1],(pl.col("X").diff(n=1)/100).alias("x_change_s"))
        data.insert_column(data.shape[1],(pl.col("X").diff(n=frames_skipped)/100).alias("x_change"))
        data.insert_column(data.shape[1],(pl.col("Y").diff(n=frames_skipped)).alias("y_change"))
        data.insert_column(data.shape[1],(pl.col("x_change")*frame_rate/frames_skipped).alias("velocity(m/s)"))
        data.insert_column(data.shape[1], pl.lit((((data["x_change_s"]-data["x_change_s"].mean())**2).sum()/data.shape[0])**0.5).alias("delta_x_change"))
        data.insert_column(data.shape[1], pl.lit(1/frame_rate*frames_skipped).alias("time_change")) ## time between frames
        data.insert_column(data.shape[1], pl.lit(shutter_speed).alias("delta_time_change")) ##shutter speed
        err_formula = formula.get_error_expr()
        clean = data.drop_nans().drop_nulls()
        evaluated= clean.with_columns(err_formula.evaluate_polars(clean).alias("err_v"))
        result = data.join(
            evaluated.select(["frame", "err_v"]),
            on="frame",
            how="left",
        )
        data = result
        data = data.filter(~pl.Series(range(len(data))).is_in([0,1]))
        # create linear fit with velocity as dependency of time.
        data = data.drop_nulls()
        time = data["time"].to_numpy()
        vel  = data["X"].to_numpy()
        Time = sm.add_constant(time)
        model = sm.OLS(vel, Time).fit()


        # save model summary to file.
        summary = model.summary(
            xname=["Intercept (x0)", "velocity (m/s)"],
            yname="X"
        )
        with open(output_folder+"reg_sum.txt", "w") as f:
            f.write(summary.as_text())
        data = data.drop_nulls()
        time = data["time"].to_numpy()
        vel  = data["velocity(m/s)"].to_numpy()
        Time = sm.add_constant(time)
        model = sm.WLS(vel, Time,weights=data["err_v"].to_numpy()).fit()


        # save model summary to file.
        summary = model.summary(
            xname=["Intercept (v0)", "velocity (m*s^-2)"],
            yname="V"
        )
        with open(output_folder+"reg_sum_v.txt", "w") as f:
            f.write(summary.as_text())

        ## create plot
        fig, ax= plt.subplots(nrows=2)
        ax[0].errorbar(x=data["time"],y=data["velocity(m/s)"],yerr=data["err_v"],fmt="ok",capsize=5.0)
        ax[0].set(xlabel = "T (s)", ylabel="V (m/s)")
        ax[0].grid()
        ax[0].set_title("Ball speed")
        ax[1].scatter(x=data["time"],y=data["X"])
        ax[1].set(xlabel = "T (s)", ylabel="X (m)")
        ax[1].grid()
        ax[1].set_title("Ball place")
        plt.tight_layout()
        fig.savefig(output_folder+"plot.svg")

        ##write to output
        data.write_csv(output_folder+"/output.csv", separator = ",")

if __name__ == "__main__":
    main()
