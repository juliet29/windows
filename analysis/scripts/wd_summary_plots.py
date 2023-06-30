# ============================================================================ #
# Window detect specific plotting functions that operacte across objects --> see summary/230530_pres and 230529_summary2.ipynb for use cases 

def make_dual_plot(time, obj1, obj2, title_arr): 
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=(title_arr[0], title_arr[1],))

    s2, s3 = [[obj.window_norm, obj.temp_norm, obj.smooth_series, obj.dif, obj.deriv, obj.deriv2] for obj in [obj1, obj2]]

    names = ["Window", "Observed Temp", "Smoothed", "Difference", "Deriv1", "Deriv2"]
    
    opacities = [0.4] + [1]*(len(s2)-1)

    for ix, name, ser in zip(range(len(names)), names, s2):
        fig.add_trace(go.Scatter(x=time, y=ser, name=name, mode='lines', line_color=h.colorway[ix], opacity=opacities[ix],legendgroup=name,), row = 1, col = 1)

    for ix, name, ser in zip(range(len(names)), names, s3):
        fig.add_trace(go.Scatter(x=time, y=ser, name=name, mode='lines', line_color=h.colorway[ix], opacity=opacities[ix],legendgroup=name, showlegend=False), row = 1, col = 2)

    return fig, names

def make_dual_plot_abstract(time, objects, names, traces, title_arr, mode="lines"): 
    fig = make_subplots(rows=1, cols=len(objects), shared_yaxes=True, subplot_titles=tuple(title_arr))
    
    series = [[getattr(obj, trace) for trace in traces] for obj in objects]


    opacities = [0.4] + [1]*(len(series[0])-1)

    for series_ix, s in enumerate(series):
        leg = True if series_ix == 0 else False
        for ix, name, ser in zip(range(len(names)), names, s):
            if mode != "lines": 
                # caveat for plotting guesses 
                time_guess = time if ix+1 != len(s) else objects[series_ix].guess_times

                fig.add_trace(go.Scatter(x=time_guess, y=ser, name=name, mode=mode[ix], line_color=h.colorway[ix], opacity=opacities[ix],legendgroup=name, showlegend=leg), row = 1, col = series_ix+1)
            else:
                fig.add_trace(go.Scatter(x=time, y=ser, name=name, mode=mode, line_color=h.colorway[ix], opacity=opacities[ix],legendgroup=name, showlegend=leg), row = 1, col = series_ix+1)

    return fig, names


def update_dual_plot(fig, names, show_arr, ):
    for name in names: 
        if name in show_arr:
            fig.update_traces(visible=True, selector=dict(name=name))
        else:
            fig.update_traces(visible="legendonly", selector=dict(name=name))
    return fig

def plot_many_dist(objects, title_arr):
    # triple plot for distributions 
    fig = make_subplots(rows=1, cols=len(objects), shared_yaxes=True, subplot_titles=title_arr)
    marker_width = 0.1 
    bin_size = 0.003

    for obj_ix, obj in enumerate(objects):
        leg = True if obj_ix == 0 else False
        for ix, ser in enumerate([obj.deriv2, obj.deriv]):
            opacity = 0.9 if ix == 0 else 1
            color = '#702632' if ix == 0 else '#A4B494'
            fig.add_trace(go.Histogram(
            x=ser, histnorm='probability', name=f' Deriv{2 - ix}', opacity=opacity, marker_line=dict(width=marker_width ,color='black'), xbins=dict(size=bin_size), marker_color=color, showlegend=leg), row = 1, col = obj_ix+1)

    fig.update_layout(barmode="stack")

    return fig