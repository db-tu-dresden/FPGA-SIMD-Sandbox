/*
###############################
## Created: Intel Corporation 
##          Christian Faerber
##          PSG CE EMEA TS-FAE 
##          June 2022
###############################
*/

`timescale 1 ps / 1 ps
 
module adder_uint_module (
  input   clock,
  input   resetn,
  input   ivalid, 
  input   iready,
  output  ovalid,
  output  oready,
  input   [31:0]  datain_a,
  input   [31:0]  datain_b,    
  output  [31:0]  dataout);
 
  assign  ovalid = 1'b1;
  assign  oready = 1'b1;
  // clk, ivalid, iready, resetn are ignored
  assign dataout = datain_a + datain_b;
 
endmodule