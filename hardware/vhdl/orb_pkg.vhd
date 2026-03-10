library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

package orb_pkg is

    -- Constants
    constant LAMBDA_2_WIDTH : integer := 16;
    constant TIMESTAMP_WIDTH : integer := 64;

    -- Orb Payload Record
    type t_orb_payload is record
        orb_id       : std_logic_vector(255 downto 0);
        lambda_2     : std_logic_vector(LAMBDA_2_WIDTH-1 downto 0);
        entropy_h    : std_logic_vector(15 downto 0);
        timestamp    : std_logic_vector(TIMESTAMP_WIDTH-1 downto 0);
        quaternion   : std_logic_vector(127 downto 0); -- w, x, y, z
        signature    : std_logic_vector(1023 downto 0); -- PQC Signature
    end record;

    -- Confinement Modes
    type t_confinement_mode is (
        INFINITE_WELL,
        FINITE_WELL,
        BARRIER,
        FREE
    );

end package orb_pkg;
