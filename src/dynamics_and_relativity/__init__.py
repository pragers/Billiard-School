import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from error_prop import Formula


def main() -> None:
    # settings
    frame_rate = 1200 # in Hz
    shutter_speed =  1/1250 #33e-6 # in s
    delta_a = 0.01 #in m
    delta_d = 0.05e-3 # in m
    radius = delta_d * np.pi # in m

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


        delim = "\t"
        if "csv" in name_split[1]:
            delim = ","
        raw_data= pl.read_csv(data_file, has_header=True, separator=delim, row_index_name= "frame")
        data= raw_data.select(["frame","X","Y","x_ref"])
        formula = Formula("2*sqrt((x_change-X_change)**2+y_change**2)/d*t_change",vars = ["x_change","t_change","d","X_change","y_change"], consts={})


        #insert columns
        # Columns with change between frames of x,y,X
        data.insert_column(1, ((pl.col("frame"))/frame_rate).alias("time"))
        data.insert_column(data.shape[1],(pl.col("X").diff(n=1)/100).alias("x_change"))
        data.insert_column(data.shape[1], (pl.col("x_ref").diff(n=1)/100).alias("X_change"))
        data.insert_column(data.shape[1],(pl.col("Y").diff(n=1)).alias("y_change"))
        data.insert_column(data.shape[1], pl.lit(1/frame_rate).alias("time_change")) ## time between frames

        # Columns with change in (angular) distance
        data.insert_column(data.shape[1],(((pl.col("x_change")-pl.col("X_change"))**2+pl.col("y_change")**2)**0.5).alias("s_d"))
        data.insert_column(data.shape[1],(pl.col("s_d")/radius).alias("theta_change"))

        # Calculate the angular velocity by dividing angle change by time change
        data.insert_column(data.shape[1],pl.col("theta_change")/pl.col("time_change").alias("omega")) ## angular velocity
        # Get cumulative sum of total angle changed.
        data.insert_column(data.shape[1], pl.col("theta_change").cum_sum().alias("theta"))

        err_formula = formula.get_error_expr().set_consts({
            "delta_x_change":delta_a,
            "delta_X_change":delta_a,
            "delta_y_change":delta_a,
            "delta_t_change":shutter_speed,
            "delta_d":delta_d})

        #use clean data to calculate the errors properly
        clean = data.drop_nans().drop_nulls()
        evaluated= clean.with_columns(err_formula.evaluate_polars(clean, result_name = "err_omega"))
        result = data.join(
            evaluated.select(["frame", "err_omega"]),
            on="frame",
            how="left",
        )
        data = result
        #data = data.filter(~pl.Series(range(len(data))).is_in([0,1]))
        # create linear fit with angle as dependency of time.
        # data = data.drop_nulls()
        # time = data["time"].to_numpy()
        # vel  = data["theta"].to_numpy()
        # Time = sm.add_constant(time)
        # model = sm.OLS(vel, Time).fit()


        # # save model summary to file.
        # summary = model.summary(
        #     xname=["Intercept (x0)", "velocity (m/s)"],
        #     yname="X"
        # )
        # with open(output_folder+"reg_sum.txt", "w") as f:
        #     f.write(summary.as_text())
        # data = data.drop_nulls()
        # time = data["time"].to_numpy()
        # vel  = data["velocity(m/s)"].to_numpy()
        # Time = sm.add_constant(time)
        # model = sm.WLS(vel, Time,weights=1/data["err_v"].to_numpy()**2).fit()


        # # save model summary to file.
        # summary = model.summary(
        #     xname=["Intercept (v0)", "velocity (m*s^-2)"],
        #     yname="V"
        # )
        # with open(output_folder+"reg_sum_v.txt", "w") as f:
        #     f.write(summary.as_text())

        ## create plot
        fig, ax= plt.subplots(nrows=2)
        ax[0].errorbar(x=data["time"],y=data["omega"],yerr=data["err_omega"],fmt="ok",capsize=5.0)
        ax[0].set(xlabel = "T (s)", ylabel="omega (rad/s)")
        ax[0].grid()
        ax[0].set_title("Angular velocity")
        ax[1].scatter(x=data["time"],y=data["theta"])
        ax[1].set(xlabel = "T (s)", ylabel="angle (rad)")
        ax[1].grid()
        ax[1].set_title("Angular displacement")
        plt.tight_layout()
        fig.savefig(output_folder+"plot.svg")
        final_omega = data["omega"].mean()
        final_err_omega = (data["err_omega"]**2).mean()
        err_data = pl.dataframe.DataFrame({"omega":final_omega,"err_omega":final_err_omega})
        if os.path.exists("./results.csv"):
            err_data = pl.concat([pl.read_csv("./results.csv"),err_data])

        err_data.write_csv("./results.csv")
        ##write to output
        data.write_csv(output_folder+"/output.csv", separator = ",")

if __name__ == "__main__":
    main()
