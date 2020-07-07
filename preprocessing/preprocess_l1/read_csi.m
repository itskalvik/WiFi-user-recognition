function csi_trace = read_csi(file_src)
    csi_trace1 = read_bf_file(file_src);
    csi_trace = csi_trace1(~cellfun('isempty',csi_trace1));
end