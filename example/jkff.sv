//`ifdef TEST
////* TEST COMMENT\

`define OK TEST
`define TRS
`define TRY
`define ADDR 32'hFF
`define ADDR (ADDR + 32'hFF00)
`define DONE 1
`define HAAD TRY
//

`ifndef TESTCODE
`define B2B_ 1
`else
`define B2B_2 0
`endif
/**/
`timescale 1ns/1ps

module CALC#(
    parameter wd=8,
    parameter init=1)
    (
input clk
,input reset
,input wire [wd-1:0]I0
,input wire [wd-1:0]I1
,output reg [wd-1:0]O
);
`ifdef SUM
wire o = I0 + I1;
`else
wire o = I0 - I1;
`endif
always @(posedge clk or negedge reset) if (~reset) O <= 'h0; else O <= o;

    endmodule
// Module thisiscomment
/*
Header Multiline Comment
Header Multiline Comment
Header Multiline Comment
//*/`ifndef JKTEST
module jjkflipflop #(parameter dw=32, parameter init=0)
(q,qbar,clk,rst,jk,test,NT);
`else
module jjkflipflop(q,qbar,clk,rst,jk,test);
`endif
	output reg q;
	output qbar;
	input clk, rst;
	input [1:0] jk;
`ifdef TEST
    output test;
    `ifndef NT
    input NT;
    `endif
`else
    output [15:0] test16;
    `ifdef OT
        input OT;
    `elsif AT 
        output AT;//
    `endif
    input test;

    `ifndef ET inout ET; `else inout ST; `endif inout LL;   
`endif //`ifdef TEST
////* TEST COMMENT
/*
Multiline Comment
Multiline Comment
Multiline Comment
//*/
`ifdef    _A1A 0`ifndef  B2B_ `ifdef B2B_2 ! `else @ `endif 1`elsif CC  2`endif  3
`else  4 `ifdef D3D  5 `endif 6`ifdef EE 7  `endif 8 `endif 
9`ifdef F4F   10 `else  11 `endif 12

	assign qbar = ~q;

	always @(posedge clk)
	begin
		if (rst)
			q <= 0;
		else
			case(jk)
				2'b00: q <= q;
				2'b01: q <= 0;
				2'b10: q <= 1;
				2'b11: q <= ~q;
			endcase
	end

`ifndef NOINST
CALC #(.wd(16), .init(0)) icalc(
    .clk (clk)
,   .reset (rst)
,   .I0({16{q}})
,   .I1({16{~q}})
`ifdef TEST
,   .O (test)
`else
,   .O (test16)
`endif// TEST
);
`else
`endif
endmodule
