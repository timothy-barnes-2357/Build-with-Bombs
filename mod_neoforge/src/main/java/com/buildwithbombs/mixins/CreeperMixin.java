package com.buildwithbombs.mixins;

import net.minecraft.sounds.SoundEvent;
import net.minecraft.sounds.SoundEvents;
import net.minecraft.world.entity.monster.Creeper;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(Creeper.class)
public class CreeperMixin {
    @Inject(method = "getDeathSound", at = @At("RETURN"), cancellable = true)
    public void getDeathSound(CallbackInfoReturnable<SoundEvent> cir) {
        cir.setReturnValue(SoundEvents.DRAGON_FIREBALL_EXPLODE);
    }
}
